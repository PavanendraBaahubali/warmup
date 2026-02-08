mod db;
use std::time::{ Instant, SystemTime, UNIX_EPOCH };
use std::thread;
use std::time::Duration;
use crossbeam_channel::{ Receiver, bounded };
use xcap::Monitor;
use sysinfo::{ CpuRefreshKind, RefreshKind, System };
use tokio;
use regex::Regex;
use std::borrow::Cow;
use once_cell::sync::Lazy; // You'll need the 'once_cell' crate
use active_win_pos_rs::get_active_window;
use fastembed::{ TextEmbedding, InitOptions, EmbeddingModel };
use image::{ DynamicImage, ImageFormat, imageops::FilterType };
use sha2::{ Sha256, Digest };
use rusqlite::Connection;
use bytemuck::cast_slice;
use inquire::Text;

use std::process::Command;
use std::thread::sleep;

use std::sync::Mutex;

use chrono::{ DateTime, Utc, Duration as CDuration };
use chrono_english::{ parse_date_string, Dialect };
use aho_corasick::AhoCorasick;
use serde::{ Serialize, Deserialize };
use reqwest::Client;

#[derive(Debug, Serialize, Deserialize)]
struct ParsedQuery {
    semantic_query: String,
    app_name: Option<String>,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct WindowMetadata {
    app_name: String,
    window_title: String,
}

#[derive(Debug)]
struct MetaData {
    window_meta: WindowMetadata,
    file_name: String,
}

// Compile regexes only ONCE globally
static CODE_INDICATORS: Lazy<Regex> = Lazy::new(|| {
    // Looks for typical code patterns: { }, trailing ;, or keywords
    Regex::new(r"(\{|\}|;|fn |let |const |impl |import )").unwrap()
});

static GARBAGE_CHARS: Lazy<Regex> = Lazy::new(|| {
    // Matches isolated noisy chars often found in OCR (e.g., "| " or " _ ")
    Regex::new(r"(?m)^\s*[|_~]\s*$").unwrap()
});

static MODEL: Lazy<Mutex<TextEmbedding>> = Lazy::new(|| {
    let mut options = InitOptions::default();

    options.model_name = EmbeddingModel::NomicEmbedTextV15;
    options.show_download_progress = true;

    Mutex::new(TextEmbedding::try_new(options).expect("Failed to init embedding model"))
});

fn smart_clean(raw_text: &str) -> Cow<str> {
    // Step 1: Heuristic - Is this code?
    // If we find 3 or more "code indicators", assume it's code.
    let matches = CODE_INDICATORS.find_iter(raw_text).count();

    if matches >= 3 {
        // IT IS CODE: Return as-is or do minimal whitespace trim
        // Using Cow::Borrowed means 0 memory allocation if we don't change it.
        return Cow::Borrowed(raw_text.trim());
    }

    // IT IS TEXT: Aggressive cleaning
    // 1. Remove OCR garbage lines
    let cleaned = GARBAGE_CHARS.replace_all(raw_text, "");

    // 2. Fix multiple spaces (e.g., "Hello    World" -> "Hello World")
    let space_fixer = Regex::new(r"\s+").unwrap();
    let final_text = space_fixer.replace_all(&cleaned, " ");

    // Using Cow::Owned because we modified the string
    Cow::Owned(final_text.into_owned())
}

async fn windows_ocr(image: &DynamicImage) -> Result<String, Box<dyn std::error::Error>> {
    use windows::Media::Ocr::OcrEngine;
    use windows::Graphics::Imaging::{ BitmapDecoder, SoftwareBitmap };
    use windows::Storage::Streams::{ DataWriter, InMemoryRandomAccessStream };
    use std::io::Cursor;

    let engine = OcrEngine::TryCreateFromUserProfileLanguages()?;

    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png)?;

    let stream: InMemoryRandomAccessStream = InMemoryRandomAccessStream::new()?;
    let writer = DataWriter::CreateDataWriter(&stream)?;
    writer.WriteBytes(&cursor.into_inner())?;
    writer.StoreAsync()?.await?;
    writer.FlushAsync()?.await?;
    stream.Seek(0)?;

    let decoder: BitmapDecoder = BitmapDecoder::CreateAsync(&stream)?.await?;
    let bitmap: SoftwareBitmap = decoder.GetSoftwareBitmapAsync()?.await?;

    let result: windows::Media::Ocr::OcrResult = engine.RecognizeAsync(&bitmap)?.await?;

    let mut full_text = String::new();
    for line in result.Lines()? {
        full_text.push_str(&line.Text()?.to_string_lossy());
        full_text.push(' ');
    }

    Ok(full_text)
}

fn compute_dhash(img: &DynamicImage) -> u64 {
    // convert to grayscale
    let gray = img.to_luma8();

    // resize to 9x8
    let small = image::imageops::resize(&gray, 9, 8, FilterType::Triangle);

    let mut hash: u64 = 0;

    for y in 0..8 {
        for x in 0..8 {
            let left = small.get_pixel(x, y)[0];
            let right = small.get_pixel(x + 1, y)[0];

            hash <<= 1;

            if left > right {
                hash |= 1;
            }
        }
    }

    hash
}
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();

    hasher.update(text.as_bytes());

    let result = hasher.finalize();

    hex::encode(result)
}

fn split_chunks(text: &str, chunk_size: usize) -> Vec<(String, usize, usize)> {
    let mut chunks = Vec::new();

    let mut start = 0;

    while start < text.len() {
        let end = (start + chunk_size).min(text.len());

        let slice = text[start..end].to_string();

        chunks.push((slice, start, end));

        start = end;
    }

    chunks
}

fn embed_text(
    conn: &Connection,
    chunk_ids: Vec<i64>,
    text_chunks: Vec<String>
) -> anyhow::Result<()> {
    // batch embedding
    let embeddings = match MODEL.lock() {
        Ok(mut model) => {
            match model.embed(text_chunks, None) {
                Ok(e) => e,

                Err(e) => {
                    println!("Embedding error: {:?}", e);
                    return Ok(());
                }
            }
        }

        Err(e) => {
            println!("MODEL mutex poisoned: {:?}", e);
            return Ok(());
        }
    };

    for (i, emb) in embeddings.iter().enumerate() {
        let chunk_id = chunk_ids[i];

        let bytes: &[u8] = cast_slice(emb.as_slice());

        println!("{}", "Inserting vec_chunks Data into Database");
        match
            conn.execute("INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)", (
                chunk_id,
                bytes,
            ))
        {
            Ok(_) => {}
            Err(e) => {
                println!("DB insert error: {:?}", e);
            }
        }
    }

    Ok(())
}

fn worker(
    rx_hot: Receiver<(DynamicImage, MetaData)>,
    dbPool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>
) {
    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();

        while let Ok((image, meta)) = rx_hot.recv() {
            println!("Processing {}", meta.file_name);

            // compute hash BEFORE OCR
            let current_hash: u64 = compute_dhash(&image);

            // threshold logic
            let mut is_similar = false;

            // get connection from pool
            let conn = match dbPool.get() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("DB connection error: {}", e);
                    continue;
                }
            };

            // fetch hashes
            let mut stmt = match conn.prepare("SELECT p_hash FROM frames") {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Prepare failed: {}", e);
                    continue;
                }
            };

            let rows = stmt.query_map([], |row| { row.get::<_, i64>(0) });

            if let Ok(rows_iter) = rows {
                for r in rows_iter {
                    if let Ok(db_hash) = r {
                        // db_hash should be u64 from DB
                        let dist = hamming_distance(current_hash, db_hash as u64);

                        // example threshold (adjust as needed)
                        if dist <= 3 {
                            is_similar = true;
                            break;
                        }
                    }
                }
            }
            //  SKIP OCR if similar
            if is_similar {
                println!("Image too similar — skipping OCR");
                continue;
            }

            //  OTHERWISE RUN OCR
            let result = rt.block_on(async { windows_ocr(&image).await });

            match result {
                Ok(text) => {
                    let cleaned_text = smart_clean(&text);

                    println!("Cleaned Text:->{}", cleaned_text);

                    // Insert frame first
                    conn.execute(
                        "INSERT INTO frames (file_path, app_name, window_title, p_hash)
                        VALUES (?1, ?2, ?3, ?4)",
                        (
                            &meta.file_name,
                            &meta.window_meta.app_name,
                            &meta.window_meta.window_title,
                            current_hash as i64,
                        )
                    ).unwrap();

                    let frame_id: i64 = conn.last_insert_rowid(); // returns i64

                    // split into chunks
                    let chunks = split_chunks(&cleaned_text, 500);
                    let mut new_chunk_ids: Vec<i64> = Vec::new();
                    let mut new_chunk_texts: Vec<String> = Vec::new();
                    for (chunk_text, start_idx, end_idx) in chunks {
                        let text_hash = hash_text(&chunk_text);

                        // check if chunk exists
                        let exists: Result<i64, _> = conn.query_row(
                            "SELECT id FROM chunks WHERE text_hash=?1 LIMIT 1",
                            [&text_hash],
                            |row| row.get(0)
                        );

                        if exists.is_ok() {
                            println!("Duplicate chunk skipped");

                            continue;
                        }

                        // insert new chunk
                        let _ = conn.execute(
                            "INSERT INTO chunks
                (frame_id, text_content, text_hash, start_char_idx, end_char_idx)
                VALUES (?1, ?2, ?3, ?4, ?5)",
                            (frame_id, &chunk_text, &text_hash, start_idx as i64, end_idx as i64)
                        );

                        let chunk_id = conn.last_insert_rowid();
                        new_chunk_ids.push(chunk_id);
                        new_chunk_texts.push(chunk_text);

                        println!("{}", "Chunk insertion is done");
                    }

                    if !new_chunk_ids.is_empty() {
                        embed_text(&conn, new_chunk_ids, new_chunk_texts);
                    }
                }

                Err(e) => println!("OCR error: {:?}", e),
            }
        }
    });
}

fn get_window_meta_data() -> WindowMetadata {
    match get_active_window() {
        Ok(window) =>
            WindowMetadata {
                app_name: window.app_name,
                window_title: window.title,
            },
        Err(()) =>
            WindowMetadata {
                app_name: "Unknown".to_string(),
                window_title: "Unknown".to_string(),
            },
    }
}

fn watcher() {
    let monitors: Vec<Monitor> = Monitor::all().expect("Failed to get monitors");

    let primary_monitor: &Monitor = monitors.first().expect("No monitor found!");

    let monitor_name: String = primary_monitor.name().unwrap_or("Unknown Display".to_string());
    println!("Capturing screen: {}", monitor_name);

    let mut sys: System = System::new_with_specifics(
        RefreshKind::everything().with_cpu(CpuRefreshKind::everything())
    );

    let (tx_hot, rx_hot) = bounded::<(DynamicImage, MetaData)>(5);

    let pool: r2d2::Pool<r2d2_sqlite::SqliteConnectionManager> = db
        ::init_pool("search_engine.db")
        .unwrap();

    worker(rx_hot, pool.clone());

    loop {
        println!("Watching screen...");
        let start = Instant::now();

        thread::sleep(Duration::from_millis(200));
        sys.refresh_cpu_all();

        let cpu_usage = sys.global_cpu_usage();

        match primary_monitor.capture_image() {
            Ok(image_buffer) => {
                let duration: Duration = start.elapsed();
                println!("Capture took: ration{:?}", duration);

                let timestamp: u128 = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                let filename = format!("watcher/{}.png", timestamp);
                let dynamic: DynamicImage = image_buffer.into();

                let window_meta: WindowMetadata = get_window_meta_data();
                let meta = MetaData {
                    file_name: filename,
                    window_meta,
                };

                if cpu_usage > 70.0 {
                    println!("{}", "CPU in heavy mode. Pushing the images in Cold Queue.");
                    match tx_hot.try_send((dynamic, meta)) {
                        Ok(_) => {} // Sent successfully
                        Err(_) => println!("Worker is too slow! Dropping frame."),
                    }
                } else {
                    // Have to process the images instantly. since we've very less load on cpu.
                    println!("CPU is on less load. we can do OCR operations instantly");

                    println!("Processing Hot Queue{}", meta.file_name);

                    // compute hash BEFORE OCR
                    let current_hash: u64 = compute_dhash(&dynamic);

                    // threshold logic
                    let mut is_similar = false;

                    // get connection from pool
                    let conn = match pool.get() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("DB connection error: {}", e);
                            continue;
                        }
                    };

                    // fetch hashes
                    let mut stmt = match conn.prepare("SELECT p_hash FROM frames") {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("Prepare failed: {}", e);
                            continue;
                        }
                    };

                    let rows = stmt.query_map([], |row| { row.get::<_, i64>(0) });

                    if let Ok(rows_iter) = rows {
                        for r in rows_iter {
                            if let Ok(db_hash) = r {
                                // db_hash should be u64 from DB
                                let dist = hamming_distance(current_hash, db_hash as u64);

                                // example threshold (adjust as needed)
                                if dist <= 3 {
                                    is_similar = true;
                                    break;
                                }
                            }
                        }
                    }
                    //  SKIP OCR if similar
                    if is_similar {
                        println!("Image too similar — skipping OCR");
                        continue;
                    }

                    let rt = tokio::runtime::Runtime::new().unwrap();

                    //  OTHERWISE RUN OCR
                    let result = rt.block_on(async { windows_ocr(&dynamic).await });

                    match result {
                        Ok(text) => {
                            let cleaned_text = smart_clean(&text);

                            println!("{}", "Inserting Frames Data into Database");
                            // Insert frame first
                            conn.execute(
                                "INSERT INTO frames (file_path, app_name, window_title, p_hash)
                        VALUES (?1, ?2, ?3, ?4)",
                                (
                                    &meta.file_name,
                                    &meta.window_meta.app_name,
                                    &meta.window_meta.window_title,
                                    current_hash as i64,
                                )
                            ).unwrap();

                            let frame_id: i64 = conn.last_insert_rowid(); // returns i64

                            // split into chunks
                            let chunks = split_chunks(&cleaned_text, 500);
                            let mut new_chunk_ids: Vec<i64> = Vec::new();
                            let mut new_chunk_texts: Vec<String> = Vec::new();
                            for (chunk_text, start_idx, end_idx) in chunks {
                                let text_hash = hash_text(&chunk_text);

                                // check if chunk exists
                                let exists: Result<i64, _> = conn.query_row(
                                    "SELECT id FROM chunks WHERE text_hash=?1 LIMIT 1",
                                    [&text_hash],
                                    |row| row.get(0)
                                );

                                if exists.is_ok() {
                                    println!("Duplicate chunk skipped");

                                    continue;
                                }

                                println!("{}", "Inserting Chunks Data into Database");
                                // insert new chunk
                                let _ = conn.execute(
                                    "INSERT INTO chunks
                (frame_id, text_content, text_hash, start_char_idx, end_char_idx)
                VALUES (?1, ?2, ?3, ?4, ?5)",
                                    (
                                        frame_id,
                                        &chunk_text,
                                        &text_hash,
                                        start_idx as i64,
                                        end_idx as i64,
                                    )
                                );

                                let chunk_id = conn.last_insert_rowid();
                                new_chunk_ids.push(chunk_id);
                                new_chunk_texts.push(chunk_text);

                                println!("{}", "Chunk insertion is done");
                            }

                            if !new_chunk_ids.is_empty() {
                                embed_text(&conn, new_chunk_ids, new_chunk_texts);
                            }
                        }

                        Err(e) => println!("OCR error: {:?}", e),
                    }
                }
            }
            Err(e) => {
                eprintln!("Capture failed: {}", e);
            }
        }

        // 4. Wait 2 seconds (0.5 FPS) so we don't kill the CPU
        thread::sleep(Duration::from_secs(2));
    }
}

fn deterministic_parse(query: &str) -> anyhow::Result<ParsedQuery> {
    let mut app_name = None;
    let mut start_time = None;
    let end_time = Some(Utc::now());

    // APP detection (aho-corasick)
    let apps = ["chrome", "vscode", "terminal"];
    let ac = AhoCorasick::new(apps)?;
    if let Some(mat) = ac.find(query) {
        app_name = Some(apps[mat.pattern()].to_string());
    }

    // TIME detection (chrono-english)
    if query.contains("ago") || query.contains("today") {
        if let Ok(dt) = parse_date_string(query, Utc::now(), Dialect::Uk) {
            start_time = Some(dt);
        }
    }

    // semantic text cleanup (simple example)
    let semantic_query = Regex::new(r"(chrome|vscode|terminal|\d+\s*minutes?\s*ago)")?
        .replace_all(query, "")
        .to_string();

    Ok(ParsedQuery {
        semantic_query,
        app_name,
        start_time,
        end_time,
    })
}

fn parser() {
    let user_query = "What article was I reading in chrome 2 minutes ago?";

    // STEP 1: deterministic parsing
    let mut parsed = deterministic_parse(user_query);

    // // STEP 2: semantic rewrite using local Phi-3
    // let rewritten = semantic_rewrite_phi3(&parsed.semantic_query).await?;
    // parsed.semantic_query = rewritten;

    // println!("Parsed Output:\n{:#?}", parsed);
}

const OLLAMA_URL: &str = "http://127.0.0.1:11434";
const MODEL_NAME: &str = "phi3";

#[derive(Debug, Deserialize)]
struct TagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    name: String,
}

#[derive(Serialize)]
struct GenerateReq {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct GenerateResp {
    response: String,
}

async fn send_to_model(prompt: String) {
    use tokio::time::{ sleep, Duration };

    let client = Client::builder()
        .timeout(Duration::from_secs(120)) // prevent hanging forever
        .build()
        .unwrap();

    let body = GenerateReq {
        model: MODEL_NAME.to_string(),
        prompt,
        stream: false, // IMPORTANT: disable streaming
    };

    let url = format!("{}/api/generate", OLLAMA_URL);

    // retry loop
    for attempt in 1..=5 {
        println!("Sending request to model (attempt {})...", attempt);

        match client.post(&url).json(&body).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    println!("Server returned error: {}", resp.status());
                    sleep(Duration::from_secs(2)).await;
                    continue;
                }

                match resp.json::<GenerateResp>().await {
                    Ok(parsed) => {
                        println!("MODEL RESPONSE:\n{}", parsed.response);
                        return;
                    }

                    Err(e) => {
                        println!("Failed to parse response: {}", e);
                    }
                }
            }

            Err(e) => {
                println!("Request failed: {:?}", e);

                // common during cold startup
            }
        }

        sleep(Duration::from_secs(3)).await;
    }

    panic!("Model request failed after retries.");
}

use std::path::Path;
use std::os::windows::process::CommandExt;
const CREATE_NO_WINDOW: u32 = 0x08000000;

use std::fs::{ File, create_dir_all };
use std::io::{ Write, BufWriter };
use futures_util::StreamExt;
use zip::ZipArchive;

#[tokio::main]
async fn main() {
    use tokio::time::{ sleep, Duration };
    // =====================================================
    // 1. Ensure Ollama runtime exists
    // =====================================================

    let ollama_path = "./runtime/ollama.exe";

    if !Path::new(ollama_path).exists() {
        println!("Ollama runtime missing. Installing silently...");

        create_dir_all("./runtime").unwrap();

        let url =
            "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip";

        println!("Downloading Ollama runtime...");

        let response = reqwest::get(url).await.unwrap();

        let total_size = response.content_length().unwrap_or(0);

        let mut stream = response.bytes_stream();

        let zip_path = "./runtime/ollama.zip";

        let file = File::create(zip_path).unwrap();
        let mut writer = BufWriter::new(file);

        let mut downloaded: u64 = 0;

        while let Some(item) = stream.next().await {
            let chunk = item.unwrap();

            writer.write_all(&chunk).unwrap();

            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let percent = ((downloaded as f64) / (total_size as f64)) * 100.0;
                print!("\rDownloading... {:.2}%", percent);
            }
        }

        println!("\nDownload complete.");

        drop(writer); // ensure file closed

        println!("Extracting...");

        let zip_file = File::open(zip_path).unwrap();

        let mut archive = ZipArchive::new(zip_file).unwrap();

        for i in 0..archive.len() {
            let mut file = archive.by_index(i).unwrap();

            let outpath = format!("./runtime/{}", file.name());

            if file.name().ends_with('/') {
                create_dir_all(&outpath).unwrap();
            } else {
                if let Some(parent) = Path::new(&outpath).parent() {
                    create_dir_all(parent).unwrap();
                }

                let mut outfile = File::create(&outpath).unwrap();

                std::io::copy(&mut file, &mut outfile).unwrap();
            }
        }

        std::fs::remove_file(zip_path).ok();

        println!("Ollama runtime installed successfully.");
    }

    // =====================================================
    // 2. Check if Ollama server already running
    // =====================================================

    let mut running = false;

    if reqwest::get("http://localhost:11434/api/tags").await.is_ok() {
        running = true;
    }

    // =====================================================
    // 3. Start Ollama silently if not running
    // =====================================================

    if !running {
        println!("Starting Ollama hidden...");

        Command::new(ollama_path)
            .arg("serve")
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .expect("Failed to start ollama");

        // wait until ready
        for _ in 0..60 {
            if reqwest::get("http://localhost:11434/api/tags").await.is_ok() {
                println!("Ollama started.");
                running = true;
                break;
            }

            sleep(Duration::from_secs(1)).await;
        }

        if !running {
            panic!("Ollama failed to start.");
        }
    } else {
        println!("Ollama already running.");
    }

    // =====================================================
    // 4. Check if phi3 model exists
    // =====================================================

    let mut model_installed = false;

    if let Ok(resp) = reqwest::get("http://localhost:11434/api/tags").await {
        if let Ok(text) = resp.text().await {
            if text.contains("phi3") {
                model_installed = true;
            }
        }
    }

    // =====================================================
    // 5. Install model silently if missing
    // =====================================================

    if !model_installed {
        println!("Pulling phi3 model silently...");

        let status = Command::new(ollama_path)
            .args(["pull", "phi3"])
            .creation_flags(CREATE_NO_WINDOW)
            .status()
            .expect("Failed to execute pull command");

        if !status.success() {
            panic!("Model pull failed.");
        }

        println!("Model installed.");
    } else {
        println!("Model already installed.");
    }

    // =====================================================
    // 6. Warmup model (CRITICAL - prevents timeout)
    // =====================================================

    println!("Warming up model...");

    let client = reqwest::Client
        ::builder()
        .timeout(Duration::from_secs(300)) // large timeout for first load
        .build()
        .unwrap();

    let warmup_body =
        serde_json::json!({
    "model": "phi3",
    "prompt": "hi",
    "stream": false
});

    let mut warmup_ok = false;

    for attempt in 1..=10 {
        println!("Warmup attempt {}", attempt);

        match client.post("http://127.0.0.1:11434/api/generate").json(&warmup_body).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("Model warmup complete.");
                warmup_ok = true;
                break;
            }

            Ok(resp) => {
                println!("Warmup failed status: {}", resp.status());
            }

            Err(e) => {
                println!("Warmup request error: {:?}", e);
            }
        }

        sleep(Duration::from_secs(3)).await;
    }

    if !warmup_ok {
        panic!("Model warmup failed.");
    }

    // =====================================================
    // DONE
    // =====================================================

    println!("✅ Ollama ready with phi3.");

    let prompt = String::from("rust personal ai search engine");
    send_to_model(prompt).await;
}
