"""Plot the fixed analysis checkpoint; no inference or training dependencies."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
a = json.loads((root/'analysis.json').read_text())
keys = list(a['groups'])
groups = list(a['groups'].values())
labels = [k.replace('/', ' ') + (' *' if g['tasks_complete'] < 50 else '') for k,g in zip(keys,groups)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold'})
fig, axes = plt.subplots(2,2,figsize=(16,11),gridspec_kw={'height_ratios':[2.1,1]},constrained_layout=True)
blue, teal, orange = '#2458A6', '#168075', '#D67530'
y=np.arange(len(keys))
ax=axes[0,0]
p1=np.array([g['pass_k_on_complete_tasks']['1'] for g in groups])*100
p8=np.array([g['pass_k_on_complete_tasks']['8'] for g in groups])*100
ax.barh(y-.17,p1,height=.31,color=blue,label='pass@1')
ax.barh(y+.17,p8,height=.31,color=teal,label='pass@8')
intervals=np.array([g['pass8_task_wilson95'] for g in groups])*100
ax.errorbar(p8,y+.17,xerr=np.maximum(0,np.array([p8-intervals[:,0],intervals[:,1]-p8])),fmt='none',ecolor='#495057',capsize=2,lw=1)
ax.set(yticks=y,yticklabels=labels,xlim=(0,106),xlabel='Teacher success on train (%)',title='Eight samples improve coverage, unevenly')
ax.invert_yaxis();ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=axes[0,1]
values=[g['sft_records'] for g in groups]
ax.barh(y,values,color=[blue if k.startswith('4') else teal if k.startswith('6') else orange for k in keys],height=.62)
for i,v in enumerate(values):ax.text(v+.7,i,str(v),va='center')
ax.set(yticks=y,yticklabels=labels,xlim=(0,54),xlabel='Distinct training tasks with a verified SFT answer',title='161 SFT examples: 65% are 4 x 4')
ax.invert_yaxis();ax.axvline(50,color='#666',ls=':',lw=1);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
ax=axes[1,0]
shares=[a['overall']['truncation_rate']*100,a['overall']['truncation_time_share']*100]
ax.barh([0,1],shares,color=orange,label='Hit 1,024-token cap')
ax.barh([0,1],[100-v for v in shares],left=shares,color='#CFD8E3',label='Other responses')
for i,v in enumerate(shares):ax.text(v/2,i,f'{v:.1f}%',ha='center',va='center',color='white',fontweight='bold')
ax.set(yticks=[0,1],yticklabels=['Share of attempts','Share of generation time'],xlim=(0,100),xlabel='Percent',title='Truncation consumes most generation time')
ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=2,frameon=False)
ax=axes[1,1]
cats=a['preferences']['negative_categories']
vals=[cats['math_with_valid_format'],cats['format_and_math'],cats['unparseable']]
ax.barh([0,1,2],vals,color=[teal,orange,'#8897A9'])
for i,v in enumerate(vals):ax.text(v+7,i,f'{v} ({v/600:.1%})',va='center')
ax.set(yticks=[0,1,2],yticklabels=['Wrong Sudoku; valid format','Math + format failure','Unparseable'],xlim=(0,510),xlabel='Rejected answers in the 600 preference pairs',title='Most DPO negatives test real constraints')
ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Qwen3.5-9B-4bit: teacher data generation, not weight training\n3,237 saved candidates | 4 November 2026, 00:29 Chile',fontsize=18,fontweight='bold')
fig.supxlabel('* 9 x 9 hard: only 4 of 50 tasks complete (37 responses saved). Other groups: 50 tasks x 8 samples.\nError bars: approximate 95% Wilson intervals for task-level pass@8. Time excludes pauses, loading and unsaved attempts.',fontsize=10,color='#4B5563')
fig.savefig(root/'findings.png',dpi=150,facecolor='white')
fig.savefig(root/'findings.svg',facecolor='white')
print(root/'findings.png')
