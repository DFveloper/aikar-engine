from pathlib import Path
import json,os,subprocess
out=Path('/home/user/aikar-engine/experiments/validation-audit-20261115')
base=json.loads((out/'chat-seed-q1.command.json').read_text())
questions=[(out/'chat-question-1.txt').read_text(),'다음 내용을 JSON 객체로 바꿔 주세요. 다른 설명이나 코드 블록 없이 JSON만 출력하세요. 이름은 민수, 나이는 29세, 거주지는 부산입니다. 키는 name, age, city를 사용하고 age는 숫자로 쓰세요.']
(out/'chat-question-2.txt').write_text(questions[1])
checkpoints={'seed':base[base.index('--qlion-adapter')+1],'step104':'/tmp/aikar-validation-checkpoints/interleaved.adapter.gguf.step104.gguf'}
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',GGML_DISABLE_VULKAN='1',GGML_CUDA_DISABLE_GRAPHS='1',GGML_CUDA_P2P='1',LD_LIBRARY_PATH='/home/user/aikar-engine/build/bin')
for name,checkpoint in checkpoints.items():
 for i,question in enumerate(questions,1):
  cmd=base.copy();cmd[cmd.index('--qlion-adapter')+1]=checkpoint;cmd[cmd.index('-p')+1]=question
  label=f'chat-{name}-q{i}';(out/(label+'.command.json')).write_text(json.dumps(cmd,ensure_ascii=False,indent=2))
  with (out/(label+'.txt')).open('w') as stdout,(out/(label+'.log')).open('w') as stderr:
   result=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr)
  print(label,'exit',result.returncode,flush=True)
  if result.returncode:raise SystemExit(result.returncode)
