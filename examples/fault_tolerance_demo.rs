use holographic_memory::*;

fn main() {
    println!("=== 全息记忆存储 - 容错性演示 ===\n");

    let config = EncodingConfig {
        fft_window_size: 256,
        overlap_ratio: 0.5,
        redundancy_level: 3,
        phase_modulation: false,
        normalize: true,
    };

    let data: Vec<f64> = (0..512).map(|i| (i as f64 * 0.02).sin() + 0.5 * (i as f64 * 0.07).cos()).collect();
    println!("输入: {} 个采样点", data.len());

    let mut encoder = FourierEncoder::new(config.clone());
    let result = encoder.encode(&data);
    let total = result.fragments.len();
    println!("编码: {} 个片段", total);

    let weaver = RedundancyWeaver::new(3);
    let woven = weaver.weave(&result.fragments);
    println!("冗余交织后: {} 个片段", woven.len());

    for &damage_pct in &[0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let remove_count = (total as f64 * damage_pct) as usize;
        let available: Vec<HologramFragment> = result.fragments.iter()
            .skip(remove_count)
            .cloned()
            .collect();

        let decoded = encoder.decode(&available, data.len());
        let mse: f64 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() / data.len() as f64;

        let integrity = IntegrityReport::new(total as u32, available.len() as u32);
        println!(
            "  损毁 {:.0}%: 可用片段={}, 损毁率={:.2}, 可恢复={}, MSE={:.2e}",
            damage_pct * 100.0,
            available.len(),
            integrity.damage_ratio,
            integrity.recovery_possible,
            mse,
        );
    }

    println!("\n=== 完成 ===");
}
