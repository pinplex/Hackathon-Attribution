import optuna

from hackathon.models import resnet

study = optuna.load_study(study_name='resnet',
                          storage='sqlite:////Net/Groups/BGI/scratch/aschall/hackathon-attribution/resnet-74099.db')

# print(study.best_params, study.best_trial.value, study.best_trial.number)

completed = filter(lambda t: t.state == optuna.trial.TrialState.COMPLETE, study.trials)
completed = sorted(completed, key=lambda t: t.value)
completed = list(completed)
print(study.trials[0].datetime_start, study.trials[-1].datetime_complete, len(study.trials), len(completed))

for i, trial in enumerate(completed):
    if i == 5:
        break
    print("#" * 40, end='\n' * 2)
    print(f"{trial.number:03d}", trial.value)

    for (n, v) in trial.params.items():
        print(f'{n:>15}: {v:02f}')

    print('')
