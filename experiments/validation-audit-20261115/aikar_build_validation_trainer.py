from pathlib import Path
import shlex,subprocess
repo=Path('/home/user/aikar-engine');out=repo/'build-validation-runtime';objdir=repo/'build-joint/examples/qlora_training/CMakeFiles/llama-finetune-qlion.dir';flags={}
for line in (objdir/'flags.make').read_text().splitlines():
 if ' = ' in line:
  k,v=line.split(' = ',1);flags[k]=shlex.split(v)
subprocess.run(['c++','-I'+str(repo/'ggml/include'),*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(repo/'examples/qlora_training'),'-I'+str(repo/'ggml/src'),'-I/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/scripts/webtext-distill','-c',str(out/'finetune-recovery.cpp'),'-o',str(out/'finetune-recovery.o')],check=True)
link=shlex.split((objdir/'link.txt').read_text());link[link.index('-o')+1]=str(out/'llama-finetune-qo')
for i,a in enumerate(link):
 if a.endswith('/finetune_qat.cpp.o'):link[i]=str(out/'finetune-recovery.o')
 if a.endswith('/libllama.so.0.7.0'):link[i]=str(out/'libllama.so.0.7.0')
 if a.endswith('/libggml-base.so'):link[i]='/mnt/openwebui/AIKAR/Lumen-3.8-Pulsar/.cache/ga16-runtime/libggml-base.so.0.27.0'
subprocess.run(link,cwd=objdir.parent.parent,check=True)
print('Built',out/'llama-finetune-qo')
