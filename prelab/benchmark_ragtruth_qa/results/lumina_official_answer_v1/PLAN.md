# LUMINA 作者原生整答聚合：附录核验

A层：本目录仅复核每答全部原始 token 的作者分数均值，固定LUMINA；IPR、负MMD为消融。B层：统一4-BPE/answermax及统一阈值仍由旧score_lumina_qa_v1.py负责，是主任务比较；旧文件不改。

主指标 AUROC、梯形积分 AUPRC、PCC；另明确列 AP。F1Opt 仅为论文式评测统计，不保存阈值，不用于选择。没有4-BPE、整答最大值、训练、调权或部署阈值。cal159仍是已使用的开发集。

prepare/check不读特征；score须3839答缓存完整。verify独立用逐列math.fsum与并列排名/计数重算。四命令均显式调用，prepare/check不自动启动score。旧计分和特征目录只读。
