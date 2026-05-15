#![allow(clippy::needless_range_loop)]

const GF_GEN: u8 = 0x03;

const fn gf_mul_const(a: u8, b: u8) -> u8 {
    let mut p: u32 = 0;
    let mut aa = a as u32;
    let mut bb = b as u32;
    let mut i = 0;
    while i < 8 {
        if bb & 1 != 0 {
            p ^= aa;
        }
        bb >>= 1;
        aa <<= 1;
        if aa & 0x100 != 0 {
            aa ^= 0x11B;
        }
        i += 1;
    }
    p as u8
}

static GF_EXP: [u8; 512] = {
    let mut table = [0u8; 512];
    let mut v: u8 = 1;
    let mut i = 0;
    while i < 255 {
        table[i] = v;
        v = gf_mul_const(v, GF_GEN);
        i += 1;
    }
    table[255] = 1;
    let mut i = 256;
    while i < 512 {
        table[i] = table[i - 255];
        i += 1;
    }
    table
};

static GF_LOG: [u8; 256] = {
    let mut table = [0u8; 256];
    table[0] = 255;
    let mut i = 0;
    while i < 255 {
        table[GF_EXP[i] as usize] = i as u8;
        i += 1;
    }
    table
};

#[inline]
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let log_sum = (GF_LOG[a as usize] as u16 + GF_LOG[b as usize] as u16) as usize;
    GF_EXP[log_sum]
}

#[inline]
#[allow(dead_code)]
fn gf_div(a: u8, b: u8) -> u8 {
    if a == 0 { return 0; }
    assert!(b != 0, "GF除法: 除数为零");
    let log_diff = (GF_LOG[a as usize] as i32 + 255 - GF_LOG[b as usize] as i32) as usize;
    GF_EXP[log_diff % 255]
}

#[inline]
fn gf_inv(a: u8) -> u8 {
    if a == 0 { panic!("GF逆元: 零无逆元"); }
    GF_EXP[255 - GF_LOG[a as usize] as usize]
}

#[inline]
fn gf_pow(a: u8, exp: u32) -> u8 {
    if exp == 0 { return 1; }
    if a == 0 { return 0; }
    let result = (GF_LOG[a as usize] as u64 * exp as u64) % 255;
    GF_EXP[result as usize]
}

#[allow(dead_code)]
fn gf_poly_eval(coeffs: &[u8], x: u8) -> u8 {
    let mut result: u8 = 0;
    for &c in coeffs.iter().rev() {
        result = gf_mul(result, x) ^ c;
    }
    result
}

#[allow(dead_code)]
fn gf_poly_scale(poly: &[u8], scalar: u8) -> Vec<u8> {
    poly.iter().map(|&c| gf_mul(c, scalar)).collect()
}

#[allow(dead_code)]
fn gf_poly_add(a: &[u8], b: &[u8]) -> Vec<u8> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0u8; max_len];
    for i in 0..a.len() {
        result[max_len - a.len() + i] ^= a[i];
    }
    for i in 0..b.len() {
        result[max_len - b.len() + i] ^= b[i];
    }
    result
}

#[allow(dead_code)]
fn gf_poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    if a.is_empty() || b.is_empty() { return vec![]; }
    let mut result = vec![0u8; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] ^= gf_mul(a[i], b[j]);
        }
    }
    result
}

#[allow(dead_code)]
fn gf_poly_div(dividend: &[u8], divisor: &[u8]) -> (Vec<u8>, Vec<u8>) {
    if dividend.len() < divisor.len() {
        return (vec![], dividend.to_vec());
    }
    let mut out = dividend.to_vec();
    let normalizer = divisor[0];
    for i in 0..(dividend.len() - divisor.len() + 1) {
        if out[i] != 0 {
            let scale = gf_div(out[i], normalizer);
            out[i] = 0;
            for j in 1..divisor.len() {
                out[i + j] ^= gf_mul(scale, divisor[j]);
            }
        }
    }
    let sep = dividend.len() - divisor.len() + 1;
    let remainder = out[sep..].to_vec();
    let quotient = out[..sep].to_vec();
    (quotient, remainder)
}

#[derive(Debug, Clone)]
pub struct ReedSolomon {
    data_shards: usize,
    parity_shards: usize,
    total_shards: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum RsError {
    #[error("碎片数不足: 需要{required}个, 实际{available}个")]
    InsufficientShards { required: usize, available: usize },
    #[error("数据碎片数为零")]
    ZeroDataShards,
    #[error("校验碎片数超出限制: 最多255个")]
    TooManyParityShards,
    #[error("数据长度不一致: 期望{expected}, 实际{got}")]
    InconsistentLength { expected: usize, got: usize },
    #[error("无效碎片索引: {0}")]
    InvalidIndex(usize),
}

impl ReedSolomon {
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, RsError> {
        if data_shards == 0 {
            return Err(RsError::ZeroDataShards);
        }
        if parity_shards == 0 {
            return Err(RsError::TooManyParityShards);
        }
        if data_shards + parity_shards > 255 {
            return Err(RsError::TooManyParityShards);
        }
        Ok(Self {
            data_shards,
            parity_shards,
            total_shards: data_shards + parity_shards,
        })
    }

    pub fn data_shards(&self) -> usize { self.data_shards }
    pub fn parity_shards(&self) -> usize { self.parity_shards }
    pub fn total_shards(&self) -> usize { self.total_shards }

    fn vandermonde_matrix(&self) -> Vec<Vec<u8>> {
        let mut matrix = vec![vec![0u8; self.total_shards]; self.total_shards];
        for i in 0..self.total_shards {
            let x = GF_EXP[i];
            for j in 0..self.total_shards {
                matrix[i][j] = gf_pow(x, j as u32);
            }
        }
        matrix
    }

    fn invert_matrix(matrix: &[Vec<u8>], size: usize) -> Option<Vec<Vec<u8>>> {
        let mut aug = vec![vec![0u8; 2 * size]; size];
        for i in 0..size {
            for j in 0..size {
                aug[i][j] = matrix[i][j];
            }
            aug[i][size + i] = 1;
        }

        for col in 0..size {
            let mut pivot_row = None;
            for row in col..size {
                if aug[row][col] != 0 {
                    pivot_row = Some(row);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            if pivot_row != col {
                let temp = aug[col].clone();
                aug[col] = aug[pivot_row].clone();
                aug[pivot_row] = temp;
            }

            let inv_pivot = gf_inv(aug[col][col]);
            for j in 0..2 * size {
                aug[col][j] = gf_mul(aug[col][j], inv_pivot);
            }

            for row in 0..size {
                if row != col && aug[row][col] != 0 {
                    let scale = aug[row][col];
                    for j in 0..2 * size {
                        aug[row][j] ^= gf_mul(scale, aug[col][j]);
                    }
                }
            }
        }

        let mut result = vec![vec![0u8; size]; size];
        for i in 0..size {
            for j in 0..size {
                result[i][j] = aug[i][size + j];
            }
        }
        Some(result)
    }

    fn encode_matrix(&self) -> Vec<Vec<u8>> {
        let vm = self.vandermonde_matrix();
        let mut top = vec![vec![0u8; self.data_shards]; self.data_shards];
        for i in 0..self.data_shards {
            for j in 0..self.data_shards {
                top[i][j] = vm[i][j];
            }
        }

        let top_inv = Self::invert_matrix(&top, self.data_shards).expect("Vandermonde子矩阵应可逆");

        let mut result = vec![vec![0u8; self.data_shards]; self.parity_shards];
        for i in 0..self.parity_shards {
            for j in 0..self.data_shards {
                let mut sum: u8 = 0;
                for k in 0..self.data_shards {
                    sum ^= gf_mul(vm[self.data_shards + i][k], top_inv[k][j]);
                }
                result[i][j] = sum;
            }
        }
        result
    }

    fn full_encoding_matrix(&self) -> Vec<Vec<u8>> {
        let parity_matrix = self.encode_matrix();
        let mut full = vec![vec![0u8; self.data_shards]; self.total_shards];
        for i in 0..self.data_shards {
            full[i][i] = 1;
        }
        for i in 0..self.parity_shards {
            for j in 0..self.data_shards {
                full[self.data_shards + i][j] = parity_matrix[i][j];
            }
        }
        full
    }

    pub fn encode(&self, data: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, RsError> {
        if data.len() != self.data_shards {
            return Err(RsError::InsufficientShards {
                required: self.data_shards,
                available: data.len(),
            });
        }

        let block_len = data[0].len();
        for shard in data.iter() {
            if shard.len() != block_len {
                return Err(RsError::InconsistentLength {
                    expected: block_len,
                    got: shard.len(),
                });
            }
        }

        let matrix = self.encode_matrix();
        let mut parity = vec![vec![0u8; block_len]; self.parity_shards];

        for p in 0..self.parity_shards {
            for byte_idx in 0..block_len {
                let mut val: u8 = 0;
                for d in 0..self.data_shards {
                    val ^= gf_mul(matrix[p][d], data[d][byte_idx]);
                }
                parity[p][byte_idx] = val;
            }
        }

        Ok(parity)
    }

    pub fn reconstruct(
        &self,
        shards: &[Option<Vec<u8>>],
    ) -> Result<Vec<Vec<u8>>, RsError> {
        let present_indices: Vec<usize> = shards
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_some())
            .map(|(i, _)| i)
            .collect();

        if present_indices.len() < self.data_shards {
            return Err(RsError::InsufficientShards {
                required: self.data_shards,
                available: present_indices.len(),
            });
        }

        let used_indices: Vec<usize> = present_indices[..self.data_shards].to_vec();
        let block_len = shards[used_indices[0]].as_ref().unwrap().len();

        let full_matrix = self.full_encoding_matrix();

        let mut sub_matrix = vec![vec![0u8; self.data_shards]; self.data_shards];
        for (row, &idx) in used_indices.iter().enumerate() {
            for col in 0..self.data_shards {
                sub_matrix[row][col] = full_matrix[idx][col];
            }
        }

        let inv = Self::invert_matrix(&sub_matrix, self.data_shards)
            .ok_or(RsError::InsufficientShards {
                required: self.data_shards,
                available: present_indices.len(),
            })?;

        let mut result = vec![vec![0u8; block_len]; self.data_shards];
        for out_row in 0..self.data_shards {
            for byte_idx in 0..block_len {
                let mut val: u8 = 0;
                for in_row in 0..self.data_shards {
                    let data_byte = shards[used_indices[in_row]].as_ref().unwrap()[byte_idx];
                    val ^= gf_mul(inv[out_row][in_row], data_byte);
                }
                result[out_row][byte_idx] = val;
            }
        }

        Ok(result)
    }

    pub fn reconstruct_data(
        &self,
        shards: &mut [Option<Vec<u8>>],
    ) -> Result<(), RsError> {
        let data = self.reconstruct(shards)?;
        for i in 0..self.data_shards {
            if shards[i].is_none() {
                shards[i] = Some(data[i].clone());
            }
        }
        Ok(())
    }

    pub fn verify(&self, data: &[Vec<u8>], parity: &[Vec<u8>]) -> bool {
        if data.len() != self.data_shards || parity.len() != self.parity_shards {
            return false;
        }
        let block_len = data[0].len();
        for shard in data.iter().chain(parity.iter()) {
            if shard.len() != block_len {
                return false;
            }
        }

        let matrix = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.encode_matrix())) {
            Ok(m) => m,
            Err(_) => return false,
        };

        for p in 0..self.parity_shards {
            for byte_idx in 0..block_len {
                let mut val: u8 = 0;
                for d in 0..self.data_shards {
                    val ^= gf_mul(matrix[p][d], data[d][byte_idx]);
                }
                if val != parity[p][byte_idx] {
                    return false;
                }
            }
        }
        true
    }

    pub fn erasure_tolerance(&self) -> usize {
        self.parity_shards
    }

    pub fn max_recoverable_damage_ratio(&self) -> f64 {
        self.parity_shards as f64 / self.total_shards as f64
    }
}

impl std::fmt::Display for ReedSolomon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reed-Solomon({}/{}, 容错{}={:.0}%)",
            self.data_shards,
            self.total_shards,
            self.parity_shards,
            self.max_recoverable_damage_ratio() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf_arithmetic() {
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(1, 5), 5);
        assert_eq!(gf_mul(2, 2), 4);
        assert_eq!(gf_div(6, 2), 3);
        assert_eq!(gf_mul(3, gf_inv(3)), 1);
    }

    #[test]
    fn test_gf_exp_log_roundtrip() {
        for i in 1u8..=254u8 {
            assert_eq!(GF_EXP[GF_LOG[i as usize] as usize], i);
        }
    }
}
