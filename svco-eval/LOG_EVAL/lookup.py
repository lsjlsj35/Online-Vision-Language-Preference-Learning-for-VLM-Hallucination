import argparse
from glob import glob
from pathlib import Path


base_path = Path(__file__).resolve().parent.parent / "LOG_EVAL"


METRIC_NUM = {
    "amber_gen": 4,
    "amber_dis": 4
}


def rqa(p, verbose=False):
    with open(Path(p) / 'realworldqa.txt') as f:
        lines = f.readlines()
    line = [l for l in lines if l.startswith("The accuracy is ")]
    score = float(line[0][len("The accuracy is "):]) * 100
    if verbose:
        print(p)
        print('  ', score)
    return score


def tqa(p, verbose=False):
    with open(Path(p) / 'textvqa.txt') as f:
        lines = f.readlines()
    line = [l for l in lines if l.startswith("Accuracy: ")]
    score = float(line[0][len("Accuracy: "):-2])
    if verbose:
        print(p)
        print('  ', score)
    return score


def lv(p, verbose=False):
    with open(Path(p) / 'llavabench.txt') as f:
        lines = f.readlines()
    line = [l for l in lines if l.startswith("all")]
    item = line[0].split(' ')[1]
    score = float(item)
    if verbose:
        print(p)
        print('  ', score)
    return score


def mmhal(p, verbose=False):
    with open(Path(p) / 'MMHal.txt') as f:
        lines = f.readlines()
    # Average score: 2.500
    # informativeness score rate: 68.750
    # Hallucination rate: 53.125
    avg = None
    hal = None
    for line in lines:
        if line.startswith("Average score:"):
            avg = float(line.split(':')[1].strip())
        if line.startswith("Hallucination rate:"):
            hal = float(line.split(':')[1].strip())
        if line.startswith("informativeness score"):
            inf = float(line.split(':')[1].strip())
    assert avg is not None and hal is not None

    return [avg, hal, inf]


def mmstar(p, verbose=False):
    with open(Path(p) /'mmstar.txt') as f:
        lines = f.readlines()
    flag = False
    for line in lines:
        if line.startswith("= OVERALL ="):
            flag = True
            continue
        if flag:
            return float(line.split('=')[-1].strip())*100
    raise RuntimeError


def amber_gen(p, verbose=False):
    with open(Path(p) / "amber_gen.txt") as f:
        lines = f.readlines()
    KEY = ["CHAIR:", "Cover:", "Hal:", "Cog:"]
    idx = 0
    score = []
    for line in lines:
        if line.startswith(KEY[idx]):
            score.append(float(line.split(KEY[idx])[-1].strip()))
            idx += 1
        if idx == 4:
            break
    a = 100 - score[0]
    assert len(score) == 4
    return [*score, 2*a*score[1]/(score[1]+a)]


def amber_dis(p, verbose=False):
    with open(Path(p) / "amber_dis.txt") as f:
        lines = f.readlines()
    KEY = ["Accuracy:", "Precision:", "Recall:", "F1:"]
    idx = 0
    score = []
    flag = False
    for line in lines:
        if flag:
            if line.startswith(KEY[idx]):
                score.append(float(line.split(KEY[idx])[-1].strip()))
                idx += 1
            if idx == 4:
                break
        elif line.startswith("Descriminative Task:"):
            flag = True
    assert len(score) == 4
    return score
        

def cvbench(p, verbose=False):
    with open(Path(p) / "cvbench.txt") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Combined Accuracy: "):
            return 100*float(line.split(':')[-1].strip())
    raise RuntimeError


def objecthal(p, verbose=False):
    with open(Path(p) / "ObjectHal.txt") as f:
        lines = f.readlines()
    # Response Hall   : 24.82
    # Object Hall     : 13.77
    # Object Correct  : 73.34
    # Object Recall   : 71.49
    # Average Length: 1
    chairs = None
    chairi = None
    for line in lines:
        if line.startswith("Response Hall"):
            chairs = float(line.split(':')[-1].strip())
        if line.startswith("Object Hall"):
            chairi = float(line.split(':')[-1].strip())
        if line.startswith("Object Correct"):
            ob_prec = float(line.split(':')[-1].strip())
        if line.startswith("Object Recall"):
            ob_rec = float(line.split(':')[-1].strip())
        if line.startswith("Average Length"):
            avg_len = float(line.split(':')[-1].strip())
    assert chairs is not None and chairi is not None
    return [chairs, 2*ob_prec*ob_rec/(ob_prec+ob_rec), avg_len]


def exp(p, verbose=False):
    AMBg = amber_gen(p)
    AMBg = [AMBg[0], AMBg[1], AMBg[-1]]
    AMBd = amber_dis(p)[-1]
    MMHAL = mmhal(p)[0]
    OBH = objecthal(p)[:2]
    LV = lv(p)
    RQA = rqa(p)
    TQA = tqa(p)
    MMSTAR = mmstar(p)
    CVB = cvbench(p)
    return [*AMBg, MMHAL, *OBH, LV, AMBd, RQA+TQA+MMSTAR+CVB]


def get_command():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="rule")
    parser.add_argument("--path", type=str)

    parser.add_argument("--ckpt-sort", action="store_true")

    args = parser.parse_args()
    task = [t.lower() for t in args.task.split(',')]
    
    if 'exp' in task:
        task = ['exp']
    else:
        if 'all' in task:
            task = ['rule', 'gpt']
        if 'gpt' in task:
            task.extend(['lv', 'mmhal', 'objecthal'])
            task.remove('gpt')
        if 'rule' in task:
            task.extend(['amber_gen', 'amber_dis', 'rqa', 'tqa', 'cvbench', 'mmstar'])
            task.remove('rule')
    task = list(set(task))
    task.sort()
    
    paths_reg = args.path.split(',')
    paths = []
    for p in paths_reg:
        paths.extend(glob(base_path + p))
    if args.ckpt_sort:
        tmp = [(int(p.split("checkpoint-")[1]), p) for p in paths]
        tmp.sort()
        paths = [p[1] for p in tmp]

    print(' '* int(len(paths[0]) * 2 / 3), end='')
    for t in task:
        if t == 'amber_gen':
            print(' A_chair  |  A_Cov  |  A_Hal  |  A_Cog  | A_gen_F1', end="  | ")
        elif t == "amber_dis":
            print(' A_acc  |  A_prec  |  A_rec  |  A_F1', end="  | ")
        elif t == "mmhal":
            print(' score  |  hal  | info', end="  | ")
        elif t == "objecthal":
            print(' chairs  |  OB_F1  | OB_avgL', end="  | ")
        elif t == "exp":
            print('  A_Cha  |   A_Cov  |   A_F1   |  MMHAL  |  OBJ_Cha  |  OBJ_F1   |   LV   | AMB_D  |  General', end='  |')
        else:
            print(f' {t}', end='  | ')
    print()
    for p in paths:
        if "checkpoint-" in p:
            print(p.rsplit('/', 1)[1], end='  | ')
        else:
            print(p.rsplit('/', 1)[1] + ' '*15, end='  | ')
        for t in task:
            try:
                res = eval(t)(p)
                if type(res) != list:
                    print(f"{res:.2f}", end=' |  ')
                else:
                    if t == "exp":
                        print(" & ".join([f'{i:.2f}' for i in res]), end=' |  ')
                    else:
                        print("  |   ".join([f'{i:.2f}' for i in res]), end=' |  ')
            except Exception as e:
                # print(e)
                for _ in range(METRIC_NUM.get(t, 1)):
                    print("ERROR", end=" |  ")
        print()


if __name__ == "__main__":
    get_command()