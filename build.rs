use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct ModelFile {
    url: String,
    path: String,
    sha256: String,
}

fn main() {
    println!("cargo:rerun-if-changed=models.json");
    let json = std::fs::read_to_string("models.json").expect("read models.json");
    let models: BTreeMap<String, Vec<ModelFile>> = serde_json::from_str(&json).expect("parse models.json");
    let mut out = String::from("pub static MODEL_FILES: &[(&str, &[ModelFileData])] = &[\n");
    for (stem, files) in models {
        out.push_str(&format!("    ({stem:?}, &[\n"));
        for file in files {
            out.push_str(&format!(
                "        ModelFileData {{ url: {:?}, path: {:?}, sha256: {:?} }},\n",
                file.url, file.path, file.sha256
            ));
        }
        out.push_str("    ]),\n");
    }
    out.push_str("];\n");
    std::fs::write(PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR")).join("models.rs"), out).expect("write models.rs");

    // exe icon, 'embed-resource' is only a build dependency on Windows
    #[cfg(windows)]
    embed_resource::compile("assets/windows.rc", embed_resource::NONE).manifest_optional().unwrap();
}
