from pathlib import Path
import sys
import gguf
sys.path.insert(0, '/home/user/aikar-engine/scripts')
from convert_gemma4_to_lumen4exp import convert
root=Path('/tmp/aikar-full-context-lumen')
root.mkdir(exist_ok=True)
src=gguf.GGUFReader('/mnt/openwebui/AIKAR/QLion2-test/artifacts/gemma4-tiny-6l-q4.gguf')
source=root/'gemma4-12l.gguf'
w=gguf.GGUFWriter(str(source),'gemma4')
for key,field in src.fields.items():
    if key.startswith('GGUF.') or key=='general.architecture': continue
    value=field.contents()
    if key=='gemma4.block_count': value=12
    if key.startswith('gemma4.') and isinstance(value, list) and len(value)==6: value=value*2
    w.add_key_value(key,value,field.types[0],field.types[-1] if len(field.types)>1 else None)
for t in src.tensors:
    w.add_tensor(t.name,t.data.copy(),raw_dtype=t.tensor_type)
    if t.name.startswith('blk.'):
        parts=t.name.split('.')
        parts[1]=str(int(parts[1])+6)
        w.add_tensor('.'.join(parts),t.data.copy(),raw_dtype=t.tensor_type)
w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
output=root/'lumen4exp-12l.gguf'
if output.exists(): output.unlink()
convert(source,output,shared_layers=1,context_length=1024)
