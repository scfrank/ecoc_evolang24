import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from run_cb_itlearning import main

condition_key = 'Labels'

# pre-run and cached outputs

# fs = {
#     'names': 'outputs/out.IL.sepspace1.twl.G26D64.C20.prune.N1000T100.svmC1.csv',
#     'codewords-open': 'outputs/out.IL.sepspace1.twcw.G26D64.C20.prune.N1000T100.svmC1.csv',
#     'codewords-closed': 'outputs/out.IL.sepspace1.twcwc.G26D64.C20.prune.N1000T100.svmC1.csv',
# }

# dfs = []
# for k, f in fs.items():
#     df = pd.read_csv(f)
#     df[key] = k
#     dfs.append(df)


# run all conditions here

clopts = {
    '--svm_C': '1.0',
    '--teach_by_codewords': False,
    '--teach_by_labels': False,
    '--coerce_cws': False,
    '--prune_classifiers': True,
    '-C': '20',
    '-D': '64',
    '-E': '200',
    '-G': '26',
    '-M': '26',
    '-N': '0.0',
    '-R': '1000',
    '-T': '100',
}


names_clopts = clopts.copy()
names_clopts['--teach_by_labels'] = True
names_clopts['-o'] = 'IL_names.csv'

codewords_open_clopts = clopts.copy()
codewords_open_clopts['--teach_by_codewords'] = True
codewords_open_clopts['-o'] = 'IL_codewords_open.csv'

codewords_closed_clopts = clopts.copy()
codewords_closed_clopts['--teach_by_codewords'] = True
codewords_closed_clopts['--coerce_cws'] = True
codewords_closed_clopts['-o'] = 'IL_codewords_closed.csv'

dfs = []

for key, clopt in {'names': names_clopts,
                   'codewords-open': codewords_open_clopts,
                   'codewords-closed': codewords_closed_clopts}.items():
    log = main(clopt)
    df = pd.read_csv(log)
    df[condition_key] = key
    dfs.append(df)

df = pd.concat(dfs)

fig, axes = plt.subplots(2, 2, figsize=(10,4))

df = df[df['generation'] < 201]

sns.lineplot(data=df, x='generation', y='clusters', hue=condition_key, ax=axes[0,0], legend=True)
sns.lineplot(data=df, x='generation', y='VM',       hue=condition_key, ax=axes[1,0], legend=False)
sns.lineplot(data=df, x='generation', y='CS', hue=condition_key, ax=axes[0,1], legend=False)  # cluster similarity
sns.lineplot(data=df, x='generation', y='FS', hue=condition_key, ax=axes[1,1], legend=False)  # feature similarity

axes[0,0].legend().set_title('')
axes[0,0].set_yscale('log')  # show small cluster numbers better
axes[0,0].set_xlabel('')
axes[0,1].set_xlabel('')
axes[1,0].set_ylabel('Cluster Quality')
axes[0,1].set_ylabel('Inter-Gen Cluster Sim.')
axes[1,1].set_ylabel('Inter-Gen Feature Sim.')

#sns.move_legend(axes[0], "upper left", bbox_to_anchor=(1, 1))
plt.savefig('IL_labelsvscodewords_prune.png', bbox_inches='tight')
plt.show()

