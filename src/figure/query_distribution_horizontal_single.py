import os
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')


plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 2.0  
plt.rcParams['axes.labelweight'] = 'bold'  


df = pd.read_json(
    "/home/liuaofan/code_nlpl/own_csn4/dataset/ds_label/own_dataset.jsonl", lines=True
)


df["function"] = df["label"].apply(lambda x: x[0])
df["class"] = df["label"].apply(lambda x: x[1])


df["function"] = df["function"].fillna("")
df["class"] = df["class"].fillna("")


df = df[(df["function"].map(len) > 0) | (df["class"].map(len) > 0)]
df["total"] = df["function"].map(len) + df["class"].map(len)


print(df[df["total"] == df["total"].max()])


df_by_query = df.groupby("problem_statement").first().reset_index()


df_by_query["function_count"] = df_by_query["function"].map(len)
df_by_query["class_count"] = df_by_query["class"].map(len)



function_data = df_by_query.sort_values(by='function_count', ascending=False)
class_data = df_by_query.sort_values(by='class_count', ascending=True)  


fig, ax1 = plt.subplots(figsize=(12, 6))

ax2 = ax1.twinx()


colors = ['
bar_width = 0.65  


bar1 = ax1.bar(
    range(len(function_data)),
    function_data["function_count"],
    bar_width,
    label="Function Count",
    color=colors[0],
    alpha=0.85,
    edgecolor='black',
    linewidth=1.2
)


bar2 = ax2.bar(
    range(len(class_data)),
    class_data["class_count"],
    bar_width,
    label="Class Count",
    color=colors[1],
    alpha=0.85,
    edgecolor='black',
    linewidth=1.2
)


ax1.set_xlabel("User Pull Request", fontsize=16, fontweight='bold', labelpad=12)
ax1.set_ylabel("Function Count", fontsize=16, fontweight='bold', labelpad=12, color=colors[0])
ax1.tick_params(axis='y', labelcolor=colors[0])


ax2.set_ylabel("Class Count", fontsize=16, fontweight='bold', labelpad=12, color=colors[1])
ax2.tick_params(axis='y', labelcolor=colors[1])


ax1.set_title("Function and Class Counts per Problem Statement", fontsize=20, fontweight='bold', pad=20)


ax1.set_xticks([])
ax1.set_xticklabels([])


ax1.set_ylim(0, function_data['function_count'].max() * 1.1)
ax2.set_ylim(0, class_data['class_count'].max() * 1.1)


ax1.grid(axis="y", linestyle='--', alpha=0.3, color='gray')
ax1.set_axisbelow(True)


for spine in ax1.spines.values():
    spine.set_linewidth(1.5)
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()


function_mean_line = ax1.axhline(y=function_data["function_count"].mean(), color='
                                linestyle='--', alpha=0.8, label='Function Mean', linewidth=2)
class_mean_line = ax2.axhline(y=class_data["class_count"].mean(), color='
                             linestyle='--', alpha=0.8, label='Class Mean', linewidth=2)


ax1.legend(lines1 + lines2 + [function_mean_line, class_mean_line],
          labels1 + labels2 + ['Function Mean', 'Class Mean'],
          title="Statistics", 
          title_fontsize=16, 
          loc="upper right", 
          fontsize=14,
          frameon=True, 
          facecolor='white', 
          edgecolor='black', 
          framealpha=1,
          bbox_to_anchor=(0.99, 0.99))


ax1.set_facecolor('white')
fig.patch.set_facecolor('white')


ax1.tick_params(direction='out', length=6, width=1.5, labelsize=14)
ax2.tick_params(direction='out', length=6, width=1.5, labelsize=14)


plt.tight_layout()


fig.savefig("src/figure/function_class_counts.pdf", format="pdf", bbox_inches="tight", dpi=600)


fig.savefig("src/figure/function_class_counts.png", dpi=600, bbox_inches="tight")


print("Function Count Mean:", function_data["function_count"].mean())
print("Function Count Std:", function_data["function_count"].std())
print("Function Count Max:", function_data["function_count"].max())
print("Function Count Min:", function_data["function_count"].min())
print("Class Count Mean:", class_data["class_count"].mean())
print("Class Count Std:", class_data["class_count"].std())
print("Class Count Max:", class_data["class_count"].max())
print("Class Count Min:", class_data["class_count"].min())