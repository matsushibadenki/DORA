// directory: rust_kernel/src
// file: lib.rs
// title: DORA Kernel (Rust Backend)
// description: R-STDP (Reward-modulated STDP) を実装。reward引数を追加し、学習の方向と強度を制御します。

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

/// R-STDP Weight Update Step
/// 
/// 報酬変調(Reward Modulation)を追加。
/// reward > 0: 強化学習 (LTP/LTDを適用)
/// reward < 0: 罰 (反転学習、または抑制)
/// reward = 0: 学習なし
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
    reward: f32, // Added: Reward Signal
) -> PyResult<&'py PyArray2<f32>> {
    let w = weights.as_array();
    let x_trace = pre_trace.as_array();
    let y_trace = post_trace.as_array();
    let x_spike = pre_spikes.as_array();
    let y_spike = post_spikes.as_array();

    let batch_size = x_trace.shape()[0] as f32;

    // 1. Calculate Delta W terms via Matrix Multiplication
    // 行列演算を使用していますが、これはバッチ処理の並列化のためであり、
    // アルゴリズム的には局所的なHepp則の総和です。
    let dw_plus = y_spike.t().dot(&x_trace);
    let dw_minus = y_trace.t().dot(&x_spike);

    // 2. Compute New Weights with Reward Modulation
    let mut new_w = Array2::<f32>::zeros(w.raw_dim());

    Zip::from(&mut new_w)
        .and(&w)
        .and(&dw_plus)
        .and(&dw_minus)
        .par_for_each(|nw, &cw, &p, &m| {
            // R-STDP Rule:
            // Delta = (LTP - LTD) * Reward
            // 報酬が正なら通常のSTDP、負なら逆効果（または抑制）
            let raw_delta = (a_plus * p - a_minus * m) / batch_size;
            let modulated_delta = raw_delta * reward;
            
            let updated = cw + learning_rate * modulated_delta;
            *nw = updated.max(w_min).min(w_max);
        });

    Ok(new_w.into_pyarray(py))
}

#[pymodule]
fn dora_kernel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lif_update_step, m)?)?;
    m.add_function(wrap_pyfunction!(stdp_weight_update, m)?)?;
    Ok(())
}