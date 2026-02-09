def make_confounder(args):
    import step.make_confounder
    if args.make_confounder_pass is True:
        step.make_confounder.run(args)


def run(args):
    make_confounder(args)