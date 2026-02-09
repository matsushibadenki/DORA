// directory: rust_kernel/src
// file: lib.rs
// title: DORA Kernel (Rust Backend)
// description: LIFニューロン計算に加え、STDP学習則の高速計算ロジックを追加。

use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

/// LIF Neuron update step
#[pyfunction]
fn lif_update_step<'py>(
    py: Python<'py>,
    input_current: PyReadonlyArray2<f32>,
    prev_mem: PyReadonlyArray2<f32>,
    decay: f32,
    threshold: f32,
) -> PyResult<(&'py PyArray2<f32>, &'py PyArray2<f32>)> {
    let input = input_current.as_array();
    let prev = prev_mem.as_array();
    let shape = input.raw_dim();
    let mut spikes = Array2::<f32>::zeros(shape);
    let mut new_mem = Array2::<f32>::zeros(shape);

    Zip::from(&mut new_mem)
        .and(&mut spikes)
        .and(&input)
        .and(&prev)
        .par_for_each(|mem_out, spike_out, &inp, &p_mem| {
            let v = p_mem * decay + inp;
            let s = if v >= threshold { 1.0 } else { 0.0 };
            *mem_out = v - (s * threshold);
            *spike_out = s;
        });

    Ok((
        spikes.into_pyarray(py),
        new_mem.into_pyarray(py),
    ))
}

/// STDP Weight Update Step
/// 
/// Python側からの入力:
///     weights: (Out, In) 現在の重み
///     pre_trace: (Batch, In) 前ニューロンのトレース
///     post_trace: (Batch, Out) 後ニューロンのトレース
///     pre_spikes: (Batch, In) 前ニューロンのスパイク
///     post_spikes: (Batch, Out) 後ニューロンのスパイク
///     ...params...
///
/// 戻り値:
///     new_weights: (Out, In) 更新後の重み
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn stdp_weight_update<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray2<f32>,
    pre_trace: PyReadonlyArray2<f32>,
    post_trace: PyReadonlyArray2<f32>,
    pre_spikes: PyReadonlyArray2<f32>,
    post_spikes: PyReadonlyArray2<f32>,
    learning_rate: f32,
    a_plus: f32,
    a_minus: f32,
    w_min: f32,
    w_max: f32,
) -> PyResult<&'py PyArray2<f32>> {
    let w = weights.as_array();
    let x_trace = pre_trace.as_array();   // (B, In)
    let y_trace = post_trace.as_array();  // (B, Out)
    let x_spike = pre_spikes.as_array();  // (B, In)
    let y_spike = post_spikes.as_array(); // (B, Out)

    let batch_size = x_trace.shape()[0] as f32;

    // 1. Calculate Delta W terms via Matrix Multiplication
    // LTP Term: Y_spike^T (Out, B) @ X_trace (B, In) -> (Out, In)
    let dw_plus = y_spike.t().dot(&x_trace);

    // LTD Term: Y_trace^T (Out, B) @ X_spike (B, In) -> (Out, In)
    let dw_minus = y_trace.t().dot(&x_spike);

    // 2. Compute New Weights (Parallelized element-wise op)
    let mut new_w = Array2::<f32>::zeros(w.raw_dim());

    Zip::from(&mut new_w)
        .and(&w)
        .and(&dw_plus)
        .and(&dw_minus)
        .par_for_each(|nw, &cw, &p, &m| {
            // dW = (A+ * dW+ - A- * dW-) / Batch
            let delta = (a_plus * p - a_minus * m) / batch_size;
            
            // W_new = clamp(W + lr * delta)
            let updated = cw + learning_rate * delta;
            
            // Clamp
            *nw = updated.max(w_min).min(w_max);
        });

    Ok(new_w.into_pyarray(py))
}

/// DORA Kernel Module Definition
#[pymodule]
fn dora_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lif_update_step, m)?)?;
    m.add_function(wrap_pyfunction!(stdp_weight_update, m)?)?;
    Ok(())
}