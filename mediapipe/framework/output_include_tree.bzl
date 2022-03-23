def _impl_output_include_tree(ctx):
    all_inputs = depset([], transitive = [ctx.attr.target[CcInfo].compilation_context.headers]).to_list()
    all_outputs = []
    for f in all_inputs:
        if f.extension not in ["h", "hh", "hpp", "inc"]: continue
        out = ctx.actions.declare_file("source/mediapipe/" + f.short_path)
        all_outputs += [out]
        ctx.actions.run_shell(
            outputs = [out],
            inputs = [f],
            arguments = [f.path, out.path],
            command = "cp $1 $2")
    return [
        DefaultInfo(files=depset(all_outputs))
    ]


output_include_tree = rule(
    implementation = _impl_output_include_tree,
    attrs = {
        "target": attr.label(mandatory = True)
    }
)