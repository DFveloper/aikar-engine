import json,shlex,subprocess
from pathlib import Path
root=Path('/home/user/aikar-engine');out=Path('/mnt/data/AIKAR/Lumen-3.8-Pulsar-training-20261116/efficiency-audit/acc');out.mkdir(exist_ok=True)
sources={}
s=(root/'ggml/src/ggml-opt.cpp').read_text()
needle='    GGML_ASSERT(ggml_backend_sched_alloc_graph(opt_ctx->backend_sched, opt_ctx->allocated_graph_copy));\n'
assert s.count(needle)==1
extra=r'''
    if (const char * directory = std::getenv("AIKAR_GRAPH_DUMP")) {
        static int sequence = 0;
        char path[1024];
        snprintf(path, sizeof(path), "%s/graph-%d.tsv", directory, sequence++);
        FILE * f = fopen(path, "w");
        GGML_ASSERT(f);
        auto * graph = opt_ctx->allocated_graph_copy;
        fprintf(f, "index\tptr\top\ttype\tbytes\tflags\tdata\tbuffer\tbuffer_type\tview\tsrc0\tsrc1\tsrc2\tne0\tne1\tne2\tne3\tname\tsources\n");
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * n = graph->nodes[i];
            fprintf(f, "%d\t%p\t%s\t%s\t%zu\t%d\t%p\t%p\t%s\t%p\t%p\t%p\t%p\t%lld\t%lld\t%lld\t%lld\t%s\t", i, (void *) n, ggml_op_name(n->op), ggml_type_name(n->type), ggml_nbytes(n), n->flags, n->data, (void *) n->buffer, n->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(n->buffer)) : "none", (void *) n->view_src, (void *) n->src[0], (void *) n->src[1], (void *) n->src[2], (long long)n->ne[0], (long long)n->ne[1], (long long)n->ne[2], (long long)n->ne[3], n->name);
            for (auto * source : n->src) fprintf(f, "%p,", (void *) source);
            fprintf(f, "\n");
        }
        fclose(f);
    }
'''
s=s.replace(needle,needle+extra)
sources['ggml/src/ggml-opt.cpp']=s
s=(root/'ggml/src/ggml-backend.cpp').read_text();s=s.replace('    int prev_backend_id = -1;\n','    int prev_backend_id = -1;\n    const bool audit = std::getenv("AIKAR_SPLIT_TIMING") != nullptr;\n')
s=s.replace('        int split_backend_id = split->backend_id;','        const int64_t audit_start = audit ? ggml_time_us() : 0;\n        int split_backend_id = split->backend_id;')
s=s.replace('        if (!sched->callback_eval) {\n            enum ggml_status ec', '        const int64_t audit_copied = audit ? ggml_time_us() : 0;\n        if (!sched->callback_eval) {\n            enum ggml_status ec')
needle='        prev_backend_id = split_backend_id;'
assert s.count(needle)==1
s=s.replace(needle,'        if (audit) fprintf(stderr, "AUDIT_SPLIT backend=%s nodes=%d inputs=%d copy_us=%lld submit_us=%lld\\n", ggml_backend_name(split_backend), compute_graph->n_nodes, split->n_inputs, (long long)(audit_copied-audit_start), (long long)(ggml_time_us()-audit_copied));\n'+needle)
sources['ggml/src/ggml-backend.cpp']=s
commands=json.loads((root/'build/compile_commands.json').read_text());objects={}
for original,s in sources.items():
 local=out/Path(original).name;local.write_text(s);entry=next(x for x in commands if x['file']==str(root/original));args=shlex.split(entry['command']);obj=out/(local.name+'.o');args[args.index('-o')+1]=str(obj);args[-1]=str(local);args.insert(1,'-I'+str((root/original).parent))
 print('compile',local.name,flush=True)
 with (out/(local.name+'.build.log')).open('w') as f:subprocess.run(args,cwd=entry['directory'],stdout=f,stderr=subprocess.STDOUT,check=True)
 objects[Path(original).name+'.o']=str(obj)
args=shlex.split((root/'build/ggml/src/CMakeFiles/ggml-base.dir/link.txt').read_text());objects['ggml-backend-meta.cpp.o']='/tmp/aikar-live-cache/candidate/meta.o';args=[objects.get(Path(x).name,x) for x in args];args[args.index('-o')+1]=str(out/'libggml-base.so.0');args=[x for x in args if not x.startswith('-Wl,--dependency-file=')]
with (out/'link.log').open('w') as f:subprocess.run(args,cwd=root/'build/ggml/src',stdout=f,stderr=subprocess.STDOUT,check=True)
print('diagnostic library built',flush=True)
