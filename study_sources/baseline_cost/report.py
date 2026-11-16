from analyze import HERE,ROOT,read,sha,write,csvread

def table(header,rows):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    v=read(HERE/'validation.json');assert v['status']=='PASS'
    for f,h in v['sources'].items():assert sha(ROOT/f)==h,f
    effects=csvread(HERE/'baseline_effects.csv');intervals=csvread(HERE/'baseline_sampling_intervals.csv')
    block={(r['minute'],r['contrast'],r['method']):r for r in intervals}
    def ci(r):return f"[{float(r['L']):.6f}, {float(r['U']):.6f}]"
    rows=[]
    for r in effects:
        seed=f"[{float(r['seed_CI_low']):.6f}, {float(r['seed_CI_high']):.6f}]" if r['seed_CI_low'] else '不适用（确定性拟合）'
        rows.append([r['minute'],r['contrast'],f"{float(r['mean_difference']):.6f}",seed,ci(block[r['minute'],r['contrast'],'7_day']),ci(block[r['minute'],r['contrast'],'league'])])
    sizes=csvread(HERE/'model_size_training_summary.csv');timing=csvread(HERE/'inference_timing_summary.csv')
    text=['# 基线比较与模型成本报告','', '2026-11-16；对应修改清单第6、7项。属于原结果已知后的补充分析，原主分析保持原口径。', '',
          '## 1. 关键基线差值及区间','',
          '差值为左模型减Gold/XP LR的自然对数log loss，负值有利于左模型。种子区间条件于固定比赛及参数；两种块抽样区间条件于已拟合模型，分别重采样完整时间块或联赛，不包含重新训练或调参变化。区间未经多重调整，无新增p值和等效判断。','',
          table(['分钟','对照','差值','训练种子95%区间','7天块95%区间','联赛块95%区间'],rows),'',
          '- 20分钟G与Gold/XP LR的平均差为+0.000916。固定比赛下的训练种子区间为正，而两种样本重抽样区间跨零。这两类区间回答不同问题，不能选择其中一类代替全部不确定性。',
          '- 30、40分钟G和Full LR均比Gold/XP LR具有更低log loss，时间块与联赛块区间均保持负方向。结果支持按切点评价扩展特征的作用，不支持“20分钟后两个经济特征与完整模型始终等效”的概括。',
          '- 10分钟G的差值也为负；Full LR的差值为正。完整表保留所有既定比较，避免把某个时点的排名推广到全部时间。',
          '- LR为确定性单次拟合，其训练种子区间留空。重复复制同一概率不能制造训练重复。跨块仍可能共享选手，不把块抽样称为完全消除了依赖。','',
          '## 2. 参数量和原最终训练记录','',
          table(['分钟','模型','参数数','关系/隐藏宽度','训练耗时中位数（秒）','实际训练轮数中位数'],
                [(r['minute'],r['model'],r['parameters'],r['branch_width'],f"{float(r['fit_seconds_median']):.2f}",r['epochs_median']) for r in sizes]),'',
          '参数量由恢复后的实际模型计算，与四切点×三模型×十种子的120条原训练记录核对。训练耗时来自已有记录，包含原运行条件和早停差异；它不构成此次统一环境的训练速度基准。参数字节数按float32权重计算，不等同于实际推理显存或检查点文件大小。','',
          'selected_search_recipe_costs.csv按最终resolution账本的模型、切点、阶段累计被调用的拟合耗时，并列复用数量。它不包含整个研究历史的所有探索，阶段final账本也不代替从最终evaluation_lock提取的检查点清单；不能将累计拟合秒数称为项目墙钟耗时。','',
          '## 3. 统一GPU推理测量','',
          'RTX 4070 SUPER，float32、eval、原确定性配置。固定20分钟、种子111001；同一输入集合、批大小1和128；预热20次、5轮各100次调用，轮转模型测量顺序，CUDA同步计时。表中范围是五轮计时范围，非统计置信区间。','',
          table(['模型','批大小','每批中位数（ms）','每批范围（ms）','折算每场（ms）'],
                [(r['model'],r['batch_size'],f"{float(r['median_ms_batch']):.4f}",f"{float(r['min_ms_batch']):.4f}–{float(r['max_ms_batch']):.4f}",f"{float(r['median_ms_match']):.4f}") for r in timing]),'',
          '输入已驻留GPU，计时不含数据下载、特征提取、词表映射、传输及服务调度。批处理折算每场时间用于理解吞吐量，不能直接解释为单请求延迟。','',
          '20分钟G为90,210参数，N为107,442参数；G参数更少，本次测量中N前向推理更快。Flat MLP为250,241参数，却具有更短的前向时间。这说明模型参数量和执行结构是两个不同成本维度。此测量对应已选配置，不是所有图模型与MLP的通用速度结论。','',
          '## 4. 与新增抽样结果一起使用','',
          '五次抽样在全部目标上均支持matched配方更低的损失；listed目标上四次matched更低、一次listed更低。后续训练范围图应同时展示五次抽样，不能继续把原单次抽样的方向反转写成稳定规律。跨抽样均值标准差与本报告的种子/样本区间是不同统计量，绘图时分别标注。','',
          '## 5. 交付与下一步','',
          '本目录包含方案、计算脚本、逐种子差值、完整区间表、模型参数和原训练记录、推理原始计时及环境。validation.json记录一致性检查和输入哈希。','',
          '下一批为清单第8—13项：效应量图、训练范围抽样图、共同队列轨迹、图结构分布、样本筛选图与类别展示方案。当前论文正文和原实验文件未修改。','']
    (HERE/'RESULTS_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    files=[p for p in HERE.iterdir() if p.is_file() and p.name!='delivery_manifest.json']
    write(HERE/'delivery_manifest.json',{'status':'PASS','files':{p.name:sha(p) for p in files}})
    print('Report and delivery manifest complete.')

if __name__=='__main__':main()
