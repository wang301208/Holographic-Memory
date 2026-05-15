use crate::foundation::config::EncodingConfig;

pub struct AdaptiveWindowSelector {
    min_window: usize,
    max_window: usize,
}

impl AdaptiveWindowSelector {
    pub fn new(min_window: usize, max_window: usize) -> Self {
        Self {
            min_window: min_window.max(64).next_power_of_two(),
            max_window: max_window.max(256).next_power_of_two(),
        }
    }

    pub fn select(&self, data: &[f64]) -> AdaptiveResult {
        if data.is_empty() {
            return AdaptiveResult {
                window_size: self.min_window,
                overlap_ratio: 0.5,
                reasoning: "空数据，使用最小窗口".to_string(),
            };
        }

        let spectral_flatness = compute_spectral_flatness(data);
        let zero_crossing_rate = compute_zero_crossing_rate(data);
        let _rms = compute_rms(data);
        let _peak_to_rms = compute_peak_to_rms(data);

        let (window, reasoning) = if spectral_flatness > 0.8 {
            (self.max_window, format!("高谱平坦度({:.2})→宽窗口捕获宽带信号", spectral_flatness))
        } else if spectral_flatness < 0.2 {
            (self.min_window, format!("低谱平坦度({:.2})→窄窗口聚焦窄带信号", spectral_flatness))
        } else {
            let mid = ((self.min_window as f64).ln() + (self.max_window as f64).ln()) / 2.0;
            let mid_window = mid.exp() as usize;
            (mid_window.next_power_of_two(), format!("中等谱平坦度({:.2})→中等窗口", spectral_flatness))
        };

        let window = window.clamp(self.min_window, self.max_window);

        let overlap_ratio = if zero_crossing_rate > 0.3 {
            0.75
        } else if zero_crossing_rate < 0.05 {
            0.25
        } else {
            0.5
        };

        AdaptiveResult {
            window_size: window,
            overlap_ratio,
            reasoning: format!("{}; ZCR={:.2}→重叠={:.0}%", reasoning, zero_crossing_rate, overlap_ratio * 100.0),
        }
    }

    pub fn select_config(&self, data: &[f64], base_config: &EncodingConfig) -> EncodingConfig {
        let result = self.select(data);
        EncodingConfig {
            fft_window_size: result.window_size,
            overlap_ratio: result.overlap_ratio,
            ..base_config.clone()
        }
    }

    pub fn analyze(&self, data: &[f64]) -> SignalAnalysis {
        if data.is_empty() {
            return SignalAnalysis {
                spectral_flatness: 0.0,
                zero_crossing_rate: 0.0,
                rms: 0.0,
                peak_to_rms: 0.0,
                suggested_window: self.min_window,
                suggested_overlap: 0.5,
            };
        }
        let sf = compute_spectral_flatness(data);
        let zcr = compute_zero_crossing_rate(data);
        let rms = compute_rms(data);
        let p2r = compute_peak_to_rms(data);
        let result = self.select(data);
        SignalAnalysis {
            spectral_flatness: sf,
            zero_crossing_rate: zcr,
            rms,
            peak_to_rms: p2r,
            suggested_window: result.window_size,
            suggested_overlap: result.overlap_ratio,
        }
    }
}

impl Default for AdaptiveWindowSelector {
    fn default() -> Self {
        Self::new(256, 4096)
    }
}

pub struct AdaptiveResult {
    pub window_size: usize,
    pub overlap_ratio: f64,
    pub reasoning: String,
}

pub struct SignalAnalysis {
    pub spectral_flatness: f64,
    pub zero_crossing_rate: f64,
    pub rms: f64,
    pub peak_to_rms: f64,
    pub suggested_window: usize,
    pub suggested_overlap: f64,
}

fn compute_spectral_flatness(data: &[f64]) -> f64 {
    if data.len() < 4 {
        return 0.5;
    }

    let n = data.len().next_power_of_two();
    let mut padded = vec![0.0f64; n];
    padded[..data.len()].copy_from_slice(data);

    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let complex_input: Vec<num_complex::Complex64> = padded.iter().map(|&x| num_complex::Complex64::new(x, 0.0)).collect();
    let mut buffer = complex_input;
    fft.process(&mut buffer);

    let half = n / 2;
    let powers: Vec<f64> = buffer[..half].iter().map(|c| c.norm_sqr() + 1e-10).collect();

    let log_mean: f64 = powers.iter().map(|p| p.ln()).sum::<f64>() / powers.len() as f64;
    let mean_log: f64 = powers.iter().sum::<f64>() / powers.len() as f64;

    if mean_log <= 0.0 {
        return 0.0;
    }
    (log_mean.exp() / mean_log).clamp(0.0, 1.0)
}

fn compute_zero_crossing_rate(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let crossings = data.windows(2)
        .filter(|w| w[0] * w[1] < 0.0)
        .count();
    crossings as f64 / (data.len() - 1) as f64
}

fn compute_rms(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    (data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64).sqrt()
}

fn compute_peak_to_rms(data: &[f64]) -> f64 {
    let rms = compute_rms(data);
    if rms < 1e-10 {
        return 0.0;
    }
    let peak = data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
    peak / rms
}
