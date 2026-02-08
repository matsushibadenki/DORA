// directory: rust_kernel/src
// file: lib.rs
// purpose: Rustカーネル v2.1 (Fix: Import Syntax & Full Parallelism)

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray2, ToPyArray};
use ndarray::{Array2, Array3, Axis, Zip};
use rayon::prelude::*; // Fixed: used '::' instead of '.'

/// LIFニューロンの並列更新カーネル
#[pyfunction]
fn update_lif_neurons<'py>(
    py: Python<'py>,
    mut v: PyReadwriteArray2<f32>,
    current: PyReadonlyArray2<f32>,
    tau: f32,
    v_th: f32,
    dt: f32
) -> &'py pyo3::types::PyAny {
    let mut v_array = v.as_array_mut();
    let current_array = current.as_array();
    let mut spikes = Array2::<f32>::zeros(v_array.raw_dim());
    
    // ndarrayの並列イテレータ
    Zip::from(&mut v_array)
        .and(&current_array)
        .and(&mut spikes)
        .par_for_each(|v_val, &i_val, s_val| {
            let decay = dt / tau;
            *v_val = *v_val * (1.0 - decay) + i_val * decay;
            if *v_val >= v_th {
                *s_val = 1.0;
                *v_val = 0.0;
            } else {
                *s_val = 0.0;
            }
        });
    spikes.to_pyarray(py).into()
}

/// Mamba Selective Scan (Batch Parallel)
/// 3次元配列 (Batch, Time, Dim) を受け取り、Batch次元を並列処理する
#[pyfunction]
fn fast_selective_scan<'py>(
    py: Python<'py>,
    u: PyReadonlyArray3<f32>,      // (Batch, L, D)
    delta: PyReadonlyArray3<f32>,  // (Batch, L, D)
    A: PyReadonlyArray2<f32>,      // (D, N) - Shared across batch
    B: PyReadonlyArray3<f32>,      // (Batch, L, N)
    C: PyReadonlyArray3<f32>,      // (Batch, L, N)
) -> &'py pyo3::types::PyAny {
    
    let u_arr = u.as_array();
    let dt_arr = delta.as_array();
    let a_arr = A.as_array();
    let b_arr = B.as_array();
    let c_arr = C.as_array();
    
    let (batch_size, len_seq, dim_d) = u_arr.dim();
    let dim_n = a_arr.shape()[1];
    
    // Rayonによるバッチ並列化
    // 各バッチの計算結果をVec<Array2>として集める
    let output_vec: Vec<Array2<f32>> = (0..batch_size).into_par_iter().map(|b_idx| {
        // バッチ b_idx の処理 (ここは個別のスレッドで走る)
        let mut y_local = Array2::<f32>::zeros((len_seq, dim_d));
        let mut h = Array2::<f32>::zeros((dim_d, dim_n));
        
        let u_b = u_arr.index_axis(Axis(0), b_idx);     // (L, D)
        let dt_b = dt_arr.index_axis(Axis(0), b_idx);   // (L, D)
        let b_b = b_arr.index_axis(Axis(0), b_idx);     // (L, N)
        let c_b = c_arr.index_axis(Axis(0), b_idx);     // (L, N)
        
        for t in 0..len_seq {
            let u_t = u_b.row(t);
            let dt_t = dt_b.row(t);
            let b_t = b_b.row(t);
            let c_t = c_b.row(t);
            let mut y_t = y_local.row_mut(t);
            
            for d in 0..dim_d {
                let dt_val = dt_t[d];
                let u_val = u_t[d];
                let mut sum_y = 0.0;
                
                for n in 0..dim_n {
                    // Mamba Discretization & State Update
                    let da = (a_arr[[d, n]] * dt_val).exp();
                    let db = dt_val * b_t[n];
                    
                    let prev_h = h[[d, n]];
                    let new_h = da * prev_h + db * u_val;
                    h[[d, n]] = new_h;
                    
                    sum_y += new_h * c_t[n];
                }
                y_t[d] = sum_y;
            }
        }
        y_local
    }).collect();

    // 結果の結合: Vec<Array2> -> Array3
    let mut y_final = Array3::<f32>::zeros((batch_size, len_seq, dim_d));
    for (b, y_res) in output_vec.into_iter().enumerate() {
        y_final.index_axis_mut(Axis(0), b).assign(&y_res);
    }
    
    y_final.to_pyarray(py).into()
}

#[pymodule]
fn dora_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update_lif_neurons, m)?)?;
    m.add_function(wrap_pyfunction!(fast_selective_scan, m)?)?;
    Ok(())
}