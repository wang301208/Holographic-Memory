use holographic_memory::{HolographicConfig, serve};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let addr = get_addr(&args);
    let window_size = get_window_size(&args);
    let redundancy = get_redundancy(&args);

    println!("全息记忆 API 服务 v{}", env!("CARGO_PKG_VERSION"));
    println!("  监听地址: {}", addr);
    println!("  FFT窗口: {}", window_size);
    println!("  冗余等级: {}", redundancy);

    let config = HolographicConfig::default();

    let rt = tokio::runtime::Runtime::new().expect("无法创建 tokio 运行时");
    rt.block_on(async {
        if let Err(e) = serve(config, &addr).await {
            eprintln!("服务错误: {}", e);
            std::process::exit(1);
        }
    });
}

fn get_addr(args: &[String]) -> String {
    for i in 0..args.len() {
        if args[i] == "--addr" && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    "0.0.0.0:8080".to_string()
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
