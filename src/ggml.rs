//! Small wrapper over the ggml copy that whisper.cpp brings along, used to run our own
//! models on the same backends

use std::ffi::c_void;
use whisper_rs::whisper_rs_sys as sys;

pub use sys::ggml_type as Type;
pub const F32: Type = sys::ggml_type_GGML_TYPE_F32;
pub const F16: Type = sys::ggml_type_GGML_TYPE_F16;

#[derive(Clone, Copy)]
pub struct T(*mut sys::ggml_tensor);

// SAFETY: ggml objects aren't tied to the thread that made them, each model (backend, weights and their tensor handles) is only ever used by one thread at a time
unsafe impl Send for T {}
unsafe impl Send for Backend {}
unsafe impl Send for Weights {}

impl T {
    pub fn ne(self, i: usize) -> i64 {
        // SAFETY: handles always point at live tensors (see module docs)
        unsafe { (*self.0).ne[i] }
    }

    pub fn nb(self, i: usize) -> usize {
        unsafe { (*self.0).nb[i] }
    }
}

fn new_ctx(tensors: usize, graph: bool) -> *mut sys::ggml_context {
    let mut mem = tensors * unsafe { sys::ggml_tensor_overhead() };
    if graph {
        mem += unsafe { sys::ggml_graph_overhead_custom(GRAPH_SIZE, false) };
    }
    let ctx = unsafe { sys::ggml_init(sys::ggml_init_params { mem_size: mem, mem_buffer: std::ptr::null_mut(), no_alloc: true }) };
    assert!(!ctx.is_null(), "ggml_init failed");
    ctx
}

const GRAPH_SIZE: usize = 16384;

/// ggml-vulkan sets up its instance and devices without locking, so backends must be created one at a time
pub static INIT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct Backend {
    raw: sys::ggml_backend_t,
    galloc: sys::ggml_gallocr_t,
    pub gpu: bool,
}

impl Backend {
    /// Vulkan device 0 when 'gpu' and one exists otherwise CPU
    pub fn new(gpu: bool) -> Self {
        let _init = INIT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut raw = std::ptr::null_mut();
        #[cfg(feature = "vulkan")]
        if gpu && unsafe { sys::ggml_backend_vk_get_device_count() } > 0 {
            raw = unsafe { sys::ggml_backend_vk_init(0) };
        }
        let on_gpu = !raw.is_null();
        let _ = gpu;
        if raw.is_null() {
            raw = unsafe { sys::ggml_backend_cpu_init() };
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
            unsafe { sys::ggml_backend_cpu_set_n_threads(raw, threads as i32) };
        }
        let galloc = unsafe { sys::ggml_gallocr_new(sys::ggml_backend_get_default_buffer_type(raw)) };
        Self { raw, galloc, gpu: on_gpu }
    }

    /// Builds a graph with 'build', computes it and returns the output as 'f32' or 'i32'
    pub fn run<R: Copy + Default>(&self, build: impl FnOnce(&Graph) -> (T, Vec<(T, Vec<u8>)>)) -> Result<Vec<R>, String> {
        let g = Graph { ctx: new_ctx(GRAPH_SIZE, true) };
        let result = (|| {
            let (out, inputs) = build(&g);
            unsafe {
                let graph = sys::ggml_new_graph_custom(g.ctx, GRAPH_SIZE, false);
                sys::ggml_build_forward_expand(graph, out.0);
                if !sys::ggml_gallocr_alloc_graph(self.galloc, graph) {
                    return Err("out of memory for the model's working buffers".to_string());
                }
                for (t, data) in &inputs {
                    sys::ggml_backend_tensor_set(t.0, data.as_ptr() as *const c_void, 0, data.len());
                }
                if sys::ggml_backend_graph_compute(self.raw, graph) != sys::ggml_status_GGML_STATUS_SUCCESS {
                    return Err("model computation failed".to_string());
                }
                let n = sys::ggml_nelements(out.0) as usize;
                assert_eq!(sys::ggml_nbytes(out.0), n * std::mem::size_of::<R>());
                let mut v = vec![R::default(); n];
                sys::ggml_backend_tensor_get(out.0, v.as_mut_ptr() as *mut c_void, 0, n * std::mem::size_of::<R>());
                Ok(v)
            }
        })();
        unsafe { sys::ggml_free(g.ctx) };
        result
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            sys::ggml_gallocr_free(self.galloc);
            sys::ggml_backend_free(self.raw);
        }
    }
}

/// Model parameters in a backend buffer, declare tensors, then 'alloc' and 'set'
pub struct Weights {
    ctx: *mut sys::ggml_context,
    buf: sys::ggml_backend_buffer_t,
}

impl Weights {
    pub fn new(max_tensors: usize) -> Self {
        Self { ctx: new_ctx(max_tensors, false), buf: std::ptr::null_mut() }
    }

    pub fn tensor(&mut self, ty: Type, ne: &[i64]) -> T {
        let t = unsafe { sys::ggml_new_tensor(self.ctx, ty, ne.len() as i32, ne.as_ptr()) };
        T(t)
    }

    pub fn alloc(&mut self, backend: &Backend) -> Result<(), String> {
        self.buf = unsafe { sys::ggml_backend_alloc_ctx_tensors(self.ctx, backend.raw) };
        if self.buf.is_null() { Err("not enough memory for the model".into()) } else { Ok(()) }
    }

    /// Uploads 'data' converted to the tensor type
    pub fn set(&self, t: T, data: &[f32]) {
        unsafe {
            assert_eq!(sys::ggml_nelements(t.0) as usize, data.len());
            match (*t.0).type_ {
                F32 => sys::ggml_backend_tensor_set(t.0, data.as_ptr() as *const c_void, 0, data.len() * 4),
                F16 => {
                    let mut h = vec![0u16; data.len()];
                    sys::ggml_fp32_to_fp16_row(data.as_ptr(), h.as_mut_ptr(), data.len() as i64);
                    sys::ggml_backend_tensor_set(t.0, h.as_ptr() as *const c_void, 0, h.len() * 2);
                }
                ty => panic!("unsupported weight type {ty}"),
            }
        }
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe {
            if !self.buf.is_null() {
                sys::ggml_backend_buffer_free(self.buf);
            }
            sys::ggml_free(self.ctx);
        }
    }
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Graph-building context, every method adds one ggml op
pub struct Graph {
    ctx: *mut sys::ggml_context,
}

impl Graph {
    pub fn input(&self, ty: Type, ne: &[i64]) -> T {
        let t = unsafe { sys::ggml_new_tensor(self.ctx, ty, ne.len() as i32, ne.as_ptr()) };
        unsafe { sys::ggml_set_input(t) };
        T(t)
    }

    // SAFETY (all ops): handles come from live contexts and ggml validates shapes
    pub fn add(&self, a: T, b: T) -> T {
        T(unsafe { sys::ggml_add(self.ctx, a.0, b.0) })
    }
    pub fn mul(&self, a: T, b: T) -> T {
        T(unsafe { sys::ggml_mul(self.ctx, a.0, b.0) })
    }
    /// 'a' is [k, n], 'b' is [k, m], result is [n, m]
    pub fn mul_mat(&self, a: T, b: T) -> T {
        T(unsafe { sys::ggml_mul_mat(self.ctx, a.0, b.0) })
    }
    pub fn norm(&self, a: T, eps: f32) -> T {
        T(unsafe { sys::ggml_norm(self.ctx, a.0, eps) })
    }
    pub fn gelu(&self, a: T) -> T {
        T(unsafe { sys::ggml_gelu_erf(self.ctx, a.0) })
    }
    pub fn relu(&self, a: T) -> T {
        T(unsafe { sys::ggml_relu(self.ctx, a.0) })
    }
    pub fn cont(&self, a: T) -> T {
        T(unsafe { sys::ggml_cont(self.ctx, a.0) })
    }
    pub fn transpose(&self, a: T) -> T {
        T(unsafe { sys::ggml_transpose(self.ctx, a.0) })
    }
    pub fn argmax(&self, a: T) -> T {
        T(unsafe { sys::ggml_argmax(self.ctx, a.0) })
    }
    pub fn concat(&self, a: T, b: T, dim: i32) -> T {
        T(unsafe { sys::ggml_concat(self.ctx, a.0, b.0, dim) })
    }
    pub fn reshape_2d(&self, a: T, ne0: i64, ne1: i64) -> T {
        T(unsafe { sys::ggml_reshape_2d(self.ctx, a.0, ne0, ne1) })
    }
    pub fn reshape_3d(&self, a: T, ne0: i64, ne1: i64, ne2: i64) -> T {
        T(unsafe { sys::ggml_reshape_3d(self.ctx, a.0, ne0, ne1, ne2) })
    }
    /// Source dimension 'i' moves to position 'ax[i]'
    pub fn permute(&self, a: T, ax: [i32; 4]) -> T {
        T(unsafe { sys::ggml_permute(self.ctx, a.0, ax[0], ax[1], ax[2], ax[3]) })
    }
    pub fn view_2d(&self, a: T, ne0: i64, ne1: i64, nb1: usize, offset: usize) -> T {
        T(unsafe { sys::ggml_view_2d(self.ctx, a.0, ne0, ne1, nb1, offset) })
    }
    pub fn view_3d(&self, a: T, ne: [i64; 3], nb1: usize, nb2: usize, offset: usize) -> T {
        T(unsafe { sys::ggml_view_3d(self.ctx, a.0, ne[0], ne[1], ne[2], nb1, nb2, offset) })
    }
    /// Zero padding at the end of each dimension
    pub fn pad(&self, a: T, p: [i32; 4]) -> T {
        T(unsafe { sys::ggml_pad(self.ctx, a.0, p[0], p[1], p[2], p[3]) })
    }
    /// 1D im2col of 'b' [len, channels] for kernel 'a' [k, channels, out]: [channels * k, out_len]
    pub fn im2col_1d(&self, a: T, b: T, stride: i32, pad: i32) -> T {
        T(unsafe { sys::ggml_im2col(self.ctx, a.0, b.0, stride, 0, pad, 0, 1, 0, false, F16) })
    }
    /// 2D convolution: kernel 'a' [kw, kh, in, out], input 'b' [w, h, in, batch]
    pub fn conv_2d(&self, a: T, b: T, stride: [i32; 2], pad: [i32; 2]) -> T {
        T(unsafe { sys::ggml_conv_2d(self.ctx, a.0, b.0, stride[0], stride[1], pad[0], pad[1], 1, 1) })
    }

    pub fn soft_max(&self, a: T, scale: f32) -> T {
        T(unsafe { sys::ggml_soft_max_ext(self.ctx, a.0, std::ptr::null_mut(), scale, 0.0) })
    }

    /// 'x * w + b' broadcast over rows (layer norm affine part, linear bias)
    pub fn affine(&self, x: T, w: Option<T>, b: Option<T>) -> T {
        let x = w.map_or(x, |w| self.mul(x, w));
        b.map_or(x, |b| self.add(x, b))
    }

    pub fn layer_norm(&self, x: T, w: T, b: T, eps: f32) -> T {
        let n = self.norm(x, eps);
        self.affine(n, Some(w), Some(b))
    }

    pub fn linear(&self, x: T, w: T, b: Option<T>) -> T {
        let y = self.mul_mat(w, x);
        self.affine(y, None, b)
    }
}
