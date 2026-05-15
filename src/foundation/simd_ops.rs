pub struct SimdOps;

impl SimdOps {
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        let len = a.len().min(b.len());
        let mut sum = 0.0f64;
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            let a0 = a[base]; let a1 = a[base + 1]; let a2 = a[base + 2]; let a3 = a[base + 3];
            let b0 = b[base]; let b1 = b[base + 1]; let b2 = b[base + 2]; let b3 = b[base + 3];
            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            sum += a[i] * b[i];
        }

        sum
    }

    pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().min(b.len());
        let mut result = vec![0.0f64; len];
        let chunks = len / 4;
        let _remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            result[base]     = a[base]     + b[base];
            result[base + 1] = a[base + 1] + b[base + 1];
            result[base + 2] = a[base + 2] + b[base + 2];
            result[base + 3] = a[base + 3] + b[base + 3];
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] + b[i];
        }

        result
    }

    pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().min(b.len());
        let mut result = vec![0.0f64; len];
        let chunks = len / 4;
        let _remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            result[base]     = a[base]     - b[base];
            result[base + 1] = a[base + 1] - b[base + 1];
            result[base + 2] = a[base + 2] - b[base + 2];
            result[base + 3] = a[base + 3] - b[base + 3];
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] - b[i];
        }

        result
    }

    pub fn scale(a: &[f64], s: f64) -> Vec<f64> {
        let len = a.len();
        let mut result = vec![0.0f64; len];
        let chunks = len / 4;
        let _remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            result[base]     = a[base]     * s;
            result[base + 1] = a[base + 1] * s;
            result[base + 2] = a[base + 2] * s;
            result[base + 3] = a[base + 3] * s;
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] * s;
        }

        result
    }

    pub fn mul_elementwise(a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().min(b.len());
        let mut result = vec![0.0f64; len];
        let chunks = len / 4;
        let _remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            result[base]     = a[base]     * b[base];
            result[base + 1] = a[base + 1] * b[base + 1];
            result[base + 2] = a[base + 2] * b[base + 2];
            result[base + 3] = a[base + 3] * b[base + 3];
        }

        for i in (chunks * 4)..len {
            result[i] = a[i] * b[i];
        }

        result
    }

    pub fn norm_squared(a: &[f64]) -> f64 {
        Self::dot_product(a, a)
    }

    pub fn norm(a: &[f64]) -> f64 {
        Self::norm_squared(a).sqrt()
    }

    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let norm_a = Self::norm(a);
        let norm_b = Self::norm(b);
        if norm_a < 1e-15 || norm_b < 1e-15 {
            return 0.0;
        }
        Self::dot_product(a, b) / (norm_a * norm_b)
    }

    pub fn hadamard_transform(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if !n.is_power_of_two() {
            let padded_n = n.next_power_of_two();
            let mut padded = vec![0.0; padded_n];
            padded[..n].copy_from_slice(data);
            return Self::hadamard_transform_impl(&padded);
        }
        Self::hadamard_transform_impl(data)
    }

    fn hadamard_transform_impl(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n == 1 {
            return data.to_vec();
        }
        let half = n / 2;
        let mut top = vec![0.0; half];
        let mut bottom = vec![0.0; half];

        for i in 0..half {
            top[i] = data[i] + data[i + half];
            bottom[i] = data[i] - data[i + half];
        }

        let top_transformed = Self::hadamard_transform_impl(&top);
        let bottom_transformed = Self::hadamard_transform_impl(&bottom);

        let mut result = Vec::with_capacity(n);
        result.extend_from_slice(&top_transformed);
        result.extend_from_slice(&bottom_transformed);
        result
    }

    pub fn inverse_hadamard(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let transformed = Self::hadamard_transform(data);
        let inv_n = 1.0 / n as f64;
        Self::scale(&transformed, inv_n)
    }

    pub fn walsh_hadamard_encode(data: &[f64], iterations: usize) -> Vec<f64> {
        let mut result = data.to_vec();
        let scale = 1.0 / (data.len() as f64).sqrt();
        for _ in 0..iterations {
            result = Self::hadamard_transform(&result);
            result = Self::scale(&result, scale);
        }
        result
    }
}

impl std::fmt::Display for SimdOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SimdOps[4x展开+Hadamard]")
    }
}
