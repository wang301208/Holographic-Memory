use std::io::{self, Read, Write};
use std::path::PathBuf;

use holographic_memory::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let cmd = &args[1];
    let result = match cmd.as_str() {
        "store" => cmd_store(&args[2..]),
        "retrieve" => cmd_retrieve(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "status" => cmd_status(&args[2..]),
        "demo" => cmd_demo(&args[2..]),
        "help" | "--help" | "-h" => { print_usage(); Ok(()) }
        "version" | "--version" | "-v" => { println!("holographic-memory v{}", VERSION); Ok(()) }
        _ => { eprintln!("未知命令: {}", cmd); print_usage(); std::process::exit(1) }
    };

    if let Err(e) = result {
        eprintln!("错误: {}", e);
        std::process::exit(1);
    }
}

fn print_usage() {
    println!("全息记忆存储 (Holographic Memory) v{}", VERSION);
    println!();
    println!("用法: holographic-memory <命令> [选项]");
    println!();
    println!("命令:");
    println!("  store <文件>        将文件内容编码为全息片段并存储");
    println!("  retrieve <文件>     从全息存储中检索并解码文件");
    println!("  search <查询>       在全息存储中搜索相似内容");
    println!("  status              显示全息存储状态信息");
    println!("  demo                运行容错性演示");
    println!("  help                显示帮助信息");
    println!("  version             显示版本号");
    println!();
    println!("选项:");
    println!("  --data-dir <路径>   指定数据目录 (默认: ./holographic_data)");
    println!("  --window <大小>     FFT窗口大小 (默认: 1024)");
    println!("  --redundancy <等级> 冗余等级 (默认: 3)");
}

fn get_data_dir(args: &[String]) -> PathBuf {
    for i in 0..args.len() {
        if args[i] == "--data-dir" && i + 1 < args.len() {
            return PathBuf::from(&args[i + 1]);
        }
    }
    PathBuf::from("./holographic_data")
}

fn get_window_size(args: &[String]) -> usize {
    for i in 0..args.len() {
        if args[i] == "--window" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<usize>() {
                return v;
            }
        }
    }
    1024
}

fn get_redundancy(args: &[String]) -> u8 {
    for i in 0..args.len() {
        if args[i] == "--redundancy" && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<u8>() {
                return v;
            }
        }
    }
    3
}

fn cmd_store(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0].starts_with("--") {
        return Err("请指定要存储的文件路径".to_string());
    }

    let file_path = &args[0];
    let data_dir = get_data_dir(args);
    let window_size = get_window_size(args);
    let redundancy = get_redundancy(args);

    let mut file = std::fs::File::open(file_path)
        .map_err(|e| format!("无法打开文件 '{}': {}", file_path, e))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|e| format!("读取文件失败: {}", e))?;

    let data: Vec<f64> = bytes.iter().map(|&b| b as f64 / 255.0).collect();
    println!("读取 {} 字节, 转换为 {} 个采样点", bytes.len(), data.len());

    let config = EncodingConfig {
        fft_window_size: window_size,
        overlap_ratio: 0.5,
        redundancy_level: redundancy,
        phase_modulation: true,
        normalize: true,
    };

    let mut encoder = FourierEncoder::new(config.clone());
    let encode_result = encoder.encode(&data);
    println!("编码完成: {} 个全息片段", encode_result.fragments.len());

    let weaver = RedundancyWeaver::new(redundancy);
    let woven = weaver.weave(&encode_result.fragments);
    println!("冗余交织后: {} 个片段", woven.len());

    let mut index = HolographicIndex::new();
    for fragment in &woven {
        index.insert((*fragment).clone());
    }

    let engine = PersistenceEngine::new(&data_dir);
    engine.save_index(&index, "main.idx")
        .map_err(|e| format!("保存失败: {}", e))?;

    println!("存储到: {}", data_dir.display());
    println!("source_hash: {}", encode_result.source_hash);
    Ok(())
}

fn cmd_retrieve(args: &[String]) -> Result<(), String> {
    let data_dir = get_data_dir(args);
    let window_size = get_window_size(args);
    let redundancy = get_redundancy(args);

    let engine = PersistenceEngine::new(&data_dir);
    let index = engine.load_index("main.idx")
        .map_err(|e| format!("加载索引失败: {}", e))?;

    let all_fragments: Vec<HologramFragment> = index.all_fragments().into_iter().cloned().collect();
    let woven = all_fragments;

    let weaver = RedundancyWeaver::new(redundancy);
    let original = weaver.unweave(&woven);
    println!("解织得到 {} 个原始片段", original.len());

    let config = EncodingConfig {
        fft_window_size: window_size,
        overlap_ratio: 0.5,
        redundancy_level: redundancy,
        phase_modulation: true,
        normalize: true,
    };

    let mut encoder = FourierEncoder::new(config);
    let decoded = encoder.decode(&original, original.len() * window_size / 2);
    println!("解码得到 {} 个采样点", decoded.len());

    let bytes: Vec<u8> = decoded.iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();

    let stdout = io::stdout();
    let mut out = stdout.lock();
    out.write_all(&bytes)
        .map_err(|e| format!("输出失败: {}", e))?;

    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.is_empty() || args[0].starts_with("--") {
        return Err("请指定搜索查询".to_string());
    }

    let query = &args[0];
    let data_dir = get_data_dir(args);

    let engine = PersistenceEngine::new(&data_dir);
    let index = engine.load_index("main.idx")
        .map_err(|e| format!("加载索引失败: {}", e))?;

    let all_fragments: Vec<HologramFragment> = index.all_fragments().into_iter().cloned().collect();
    println!("索引中共 {} 个片段", all_fragments.len());

    let query_data: Vec<f64> = query.as_bytes().iter().map(|&b| b as f64 / 255.0).collect();
    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 2,
        phase_modulation: false,
        normalize: true,
    };
    let mut encoder = FourierEncoder::new(config);
    let query_result = encoder.encode(&query_data);

    if query_result.fragments.is_empty() {
        println!("查询编码结果为空");
        return Ok(());
    }

    let matcher = SimilarityMatcher::new(0.0);
    let results = matcher.find_similar(&query_result.fragments[0], &all_fragments, 10);

    println!("搜索结果 (Top {}):", results.len());
    for (i, item) in results.iter().enumerate() {
        println!("  #{}: 片段id={}, 相似度={:.4}", i + 1, item.fragment_id, item.similarity);
    }

    Ok(())
}

fn cmd_status(args: &[String]) -> Result<(), String> {
    let data_dir = get_data_dir(args);

    println!("全息记忆存储状态");
    println!("  数据目录: {}", data_dir.display());

    let engine = PersistenceEngine::new(&data_dir);
    match engine.load_index("main.idx") {
        Ok(index) => {
            println!("  索引状态: 已加载");
            println!("  总片段数: {}", index.len());
            let sources = index.all_source_hashes();
            println!("  数据源数: {}", sources.len());
            for &source in &sources {
                let integrity = index.integrity_check(source);
                println!("    源 {}: 总={}, 可用={}, 损毁率={:.2}, 可恢复={}",
                    source, integrity.fragments_total, integrity.fragments_available,
                    integrity.damage_ratio, integrity.recovery_possible);
            }
        }
        Err(_) => {
            println!("  索引状态: 未找到（请先使用 store 命令存储数据）");
        }
    }

    Ok(())
}

fn cmd_demo(_args: &[String]) -> Result<(), String> {
    println!("=== 全息记忆存储 - 容错性演示 ===\n");

    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: false,
        normalize: true,
    };

    let data: Vec<f64> = (0..512).map(|i| {
        let t = i as f64 * 0.02;
        t.sin() + 0.5 * (t * 3.0).cos()
    }).collect();
    println!("输入: {} 个采样点（多频信号）", data.len());

    let mut encoder = FourierEncoder::new(config.clone());
    let result = encoder.encode(&data);
    let total = result.fragments.len();
    println!("编码: {} 个片段\n", total);

    println!("{:-40}{:-12}{:-12}{:-12}{:-12}", "损毁比例", "可用片段", "可恢复", "MSE", "损坏率");
    for &pct in &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let remove = (total as f64 * pct) as usize;
        let available: Vec<HologramFragment> = result.fragments.iter()
            .skip(remove).cloned().collect();
        let decoded = encoder.decode(&available, data.len());
        let mse: f64 = data[32..480].iter().zip(decoded[32..480].iter())
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / 448.0;
        let integrity = IntegrityReport::new(total as u32, available.len() as u32);
        println!("{:-40}{:-12}{:-12}{:-12}{:-12}",
            format!("{:.0}%", pct * 100.0),
            available.len(),
            integrity.recovery_possible,
            format!("{:.2e}", mse),
            format!("{:.2}", integrity.damage_ratio),
        );
    }

    println!("\n=== 演示完成 ===");
    Ok(())
}
