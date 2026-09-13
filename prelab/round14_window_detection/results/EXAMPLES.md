# 4词元窗口的实际标记样例

粗体表示模型标出的可疑片段，不代表片段内每个词元都错误。样例在完成评分后，按问题ID顺序各取一个命中、误报和未报警案例；不能用这三例估计总体效果。

## 命中风险片段

问题：How does Lithuania’s National Agency for Education seek to drive innovation under its expanded role from July 2023?

模型标记：Under its expanded role from July 2023, Lithuania’s National Agency for Education seeks **to drive innovation by transitioning to a public administration body focused on educational reform and development.**

原标注风险片段：`by transitioning to a public administration body`；`focused on educational reform and development`

当前资料：

- Laces BC：In late November 2024, Kelsey Plum announced via social media that she would no longer be playing in the 2025 season and instead would "take some more time for myself this offseason," leaving the Laces with a wildcard spot.
- National Agency for Education (Lithuania)：The Agency officially began its work on 1 September 2019, following the merger of six institutions which were National Examination Centre, National School Evaluation Agency, Centre for Special Pedagogy and Psychology, Education Supply Centre, Centre for Information Technology in Education, Centre for The Development of Education. On 1 July 2023, the Agency's legal status was updated, transitioning from an educational support institution to a public administration body.

样本ID：`ragognize_test_0059__partial__1`

## 正常回答被误报

问题：In which city does the Turkish para-alpine skier Harun Mut live?

模型标记：Harun **Mut lives in a city not specified in the given information, but he competes for Depsaş** Enerji Sports Club.

原标注风险片段：无

当前资料：

- Abrar Fahad：Abrar Fahad was born on February 12, 1998, in Kushtia District, Bangladesh. His ancestral home was in Kumarkhali Upazila. His father, Md. Barkat Ullah, worked as an auditor at BRAC, and his mother, Rokeya Khatun, was a kindergarten teacher.
- Harun Mut：Harun Mut (born 2000) is a Turkish para-alpine skier who competes in the LW2 disability class, and an amputee football player. Mut is a member of Depsaş Enerji Sports Club.

样本ID：`ragognize_test_0648__partial__1`

## 风险回答未报警

此例按现有金标计为漏检，但Round13独立盲审与旧标签有分歧：旧标签计入问句中隐含的Nebula公司关系，盲审只判断答案字面已有资料支持。它展示了现有评测口径的局限，不能当作无争议的模型漏检。

问题：What did Jennifer Freed help develop for the astrology company Nebula?

模型标记：Jennifer Freed helped develop AI GUIDES, a personal tarot and astrology reader.

原标注风险片段：`AI GUIDES`

当前资料：

- Jennifer Freed：Jennifer Freed is an American professor, psychological astrologer and author. She has been instrumental in developing AI GUIDES, a personal tarot and astrology reader designed to offer everyday inspiration.
- Unreleased Sonic the Hedgehog games：In 2024, Iizuka said that while Sonic Team had no plans for Adventure 3, they would "love to make it".

样本ID：`ragognize_train_0043__partial__1`

