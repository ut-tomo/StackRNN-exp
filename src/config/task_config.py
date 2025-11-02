TASK_CONFIGS = {
    1: {
        'name': 'Basic Counting (a^nb^n...)',
        'nchar': 2, # TODO: make this optional
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': False,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    2: {
        'name': 'Counting + Multiplication (a^nb^kn)',
        'nchar': 2,
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': False,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    3: {
        'name': 'Counging + Addition (a^nb^mc^{n+m})',
        'nchar': 3,
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': False,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    4: {
        'name': 'Memorization (Reverse Copy)',
        'nchar': 6, # セパレータ含む, TODO: make this optional
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': True,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    5: {
        'name': 'Counting + Multiplication2 (a^nb^m c^{nm})',
        'nchar': 3,
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': False,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    6: {
        'name': 'Interleaved Counting (a^nb^mc^nd^m)',
        'nchar': 4,
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': False,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
    7: {
        'name': 'Binary Addition',
        'nchar': 5, # 0, 1, +, -, .の5種類
        'nhid': 100,
        'nstack': 10,
        'depth': 2,
        'use_noop': True,
        'bptt': 50,
        'mod': 0,
        'reg': 0.0,
    },
}

def get_task_config(task_id):
    if task_id not in TASK_CONFIGS:
        raise ValueError(f"Invalid task_id: {task_id}. Must be in range 1-7.")
    return TASK_CONFIGS[task_id].copy()

def print_task_info(task_id):
    config = get_task_config(task_id)
    print(f"\n{'='*70}")
    print(f"Task {task_id}: {config['name']}")
    print(f"{'='*70}")

    print(f"\nModel Configuration:")
    print(f"  - nchar (vocab size): {config['nchar']}")
    print(f"  - nhid (hidden size): {config['nhid']}")
    print(f"  - nstack (num stacks): {config['nstack']}")
    print(f"  - depth (stack depth): {config['depth']}")
    print(f"  - use_noop: {config['use_noop']}")
    print(f"  - mod (recurrence): {config['mod']}")
    print(f"{'='*70}\n")