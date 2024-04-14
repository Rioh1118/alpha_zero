extern crate cpython;
use cpython::{Python, NoArgs, ObjectProtocol, PyResult};

fn call_dual_network() -> PyResult<()> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let sys = py.import("sys")?;
    sys.get(py, "path")?.call_method(py, "append", ("./src/python/",), None)?;

    let dual_network = py.import("dual_network")?;

    dual_network.call(py, "dual_network", NoArgs, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_dual_network() {
        call_dual_network().unwrap();
    }
}