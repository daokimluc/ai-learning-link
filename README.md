# What to learn in AI

Viết bằng tiếng Việt. Học gì trước, cái gì để sau, mỗi link một dòng để mở hoặc copy.

# Lộ trình AI cho người mới

Mình gom lại bộ tài liệu này theo thứ tự một người mới có thể đi được. Không cần toán cao. Biết dùng máy tính, mỗi tuần dành ra vài giờ, là bắt đầu được.

Agent để ở cuối. Mới vào đã đụng agent thì dễ rối, vì agent chỉ là model cộng thêm bước gọi công cụ. Chưa hình dung model làm gì thì thêm bước đó không giúp hơn.

## Cách dùng

Ít thời gian thì làm [Lối tắt](#lối-tắt): tám mục, xong thì dừng. Muốn đi chậm hơn thì theo từ [chặng 1](#chặng-1--hiểu-llm-là-gì) đến chặng 8.

Mỗi mục có vài câu nói tại sao nên xem. Link nằm riêng một dòng, bấm hoặc copy được.

Học xong Python thì chọn một nhánh. Một nhánh là làm sản phẩm AI. Nhánh kia là làm phân tích dữ liệu. Không cần học cả hai cùng lúc.

## Cái gì mất tiền

Xem miễn phí được gần hết: video, bài báo, bài hướng dẫn, repo GitHub, và sách [Understanding Deep Learning](https://udlbook.github.io/udlbook/).

Sách của Manning và O'Reilly phải mua. Chứng chỉ thường cũng mất phí nếu muốn lấy bằng. Chỉ cần hiểu bài thì phần miễn phí với các repo là đủ.

Link khóa học ở dưới là link giới thiệu, lấy từ bài tổng hợp gốc.

## Mục lục

- [Lối tắt](#lối-tắt)
- [Chặng 1. Hiểu LLM là gì](#chặng-1--hiểu-llm-là-gì)
- [Chặng 2. Python vừa đủ](#chặng-2--python-vừa-đủ)
- [Chặng 3. Chọn một nhánh](#chặng-3--chọn-một-nhánh)
- [Chặng 4. Machine learning](#chặng-4--machine-learning)
- [Chặng 5. Deep learning](#chặng-5--deep-learning)
- [Chặng 6. Làm việc với LLM](#chặng-6--làm-việc-với-llm)
- [Chặng 7. Bốn bài báo](#chặng-7--bốn-bài-báo)
- [Chặng 8. Xây agent](#chặng-8--xây-agent)
- [Nên bỏ những gì](#nên-bỏ-những-gì)
- [Lịch 6 tuần](#lịch-6-tuần)
- [Tất cả link](#tất-cả-link)

---

## Lối tắt

Khoảng 15–20 giờ nếu làm lần lượt rồi dừng. Lúc này mở thêm nhiều tutorial thường không hơn.

Mục tiêu nhỏ thôi: tự chạy được một vòng lặp đơn giản trên máy mình.

### 1. LLM Introduction

Andrej Karpathy, video khoảng 1 giờ.

Xem cái này trước. Trong video có token là gì, model được train ra sao, và vì sao nó hay nói sai mà vẫn rất tự tin.

https://www.youtube.com/watch?v=zjkBMFhNj_g

### 2. Crash Course on Python

Khóa ngắn trên Coursera.

Đủ để đọc code người khác và sửa một dòng. Viết Python đều rồi thì bỏ qua.

https://imp.i384100.net/QYd9Dz

### 3. Machine Learning Specialization

Andrew Ng, DeepLearning.AI.

Khóa này giúp phân biệt model đang học một quy luật, hay chỉ nhớ bài đã gặp. Lối tắt mà chỉ giữ một khóa học thì giữ khóa này.

https://imp.i384100.net/k42K6M

### 4. Prompt Engineering Guide

Repo của DAIR.AI.

Họ ghi cách ra lệnh cho model, từ câu đơn giản đến vài kỹ thuật hay bị gọi là agent. Đọc phần đang cần, không cần thuộc.

https://github.com/dair-ai/Prompt-Engineering-Guide

### 5. Building Effective Agents

Bài của Anthropic.

Nên đọc trước khi làm agent. Nhiều việc giải được bằng các bước cố định. Agent chỉ đáng thêm khi cách đó không chạy.

https://www.anthropic.com/engineering/building-effective-agents

### 6. AI Agents for Beginners

Repo của Microsoft, bài học xếp sẵn thứ tự.

Không muốn tự chọn giữa hàng chục repo thì làm theo repo này.

https://github.com/microsoft/ai-agents-for-beginners

### 7. ReAct

Bài báo của Yao và cộng sự, 2022.

Vòng đầu đọc phần tóm tắt và xem hình là đủ. Ý chính: nghĩ, gọi một công cụ, nhìn kết quả, rồi nghĩ tiếp.

https://arxiv.org/abs/2210.03629

### 8. Building an Agent from Scratch

Video của Kam Lasater, khoảng 19 phút.

Mở máy và gõ theo. Chỉ xem thì tuần sau quên hết.

https://www.youtube.com/watch?v=xzXdLRUyjUg

---

## Chặng 1 — Hiểu LLM là gì

Một buổi là đủ.

Ngày đầu đừng mở repo agent. Xem một video, rồi thử nói lại bằng lời của mình: token là gì, và vì sao model cứ đoán từ tiếp theo.

### LLM Introduction

Andrej Karpathy, khoảng 1 giờ.

https://www.youtube.com/watch?v=zjkBMFhNj_g

## Chặng 2 — Python vừa đủ

Vài ngày, người chậm hơn thì gần hai tuần. Biết Python rồi thì sang chặng sau.

Chưa cần thành lập trình viên. Cần mở một file, sửa một dòng, chạy lại, và đọc được dòng lỗi.

Tạm ổn khi bạn viết được một script nhỏ: đọc một file, in ra một kết quả, và tự sửa lúc nó lỗi.

### Crash Course on Python

Khóa này đủ để đỡ ngại khi gặp code.

https://imp.i384100.net/QYd9Dz

### Python for Data Science, AI & Development

Làm quen bảng, file, và vài thư viện sẽ gặp lại trong notebook. Dùng Jupyter quen rồi thì bỏ.

https://imp.i384100.net/9Vd6kQ

## Chặng 3 — Chọn một nhánh

Excel và bài báo ReAct là hai việc khác nhau. Chọn một. Nhánh còn lại để đó, không phải bài về nhà.

### Nhánh A — muốn xây sản phẩm AI

Sang chặng 4. Nên học thêm một chút SQL, để đỡ lạc khi dữ liệu nằm trong bảng.

**SQL Foundations**

https://imp.i384100.net/k42K6v

### Nhánh B — muốn làm phân tích dữ liệu

Nhánh này để xin việc phân tích. Không phải bước bắt buộc trước khi làm agent. Không cần học hết. Một bộ vừa sức:

1. Excel cơ bản
2. Một chứng chỉ phân tích, Google hoặc Meta, chọn một
3. Power BI, chỉ khi chỗ làm đang dùng
4. ChatGPT với Excel, chỉ sau khi tự làm được một bảng

**Introduction to Data Analytics**

Khóa nhập môn: câu hỏi cần trả lời là gì, dữ liệu bẩn trông thế nào, và kể một kết quả ra sao.

https://imp.i384100.net/5k0drn

**Excel Basics for Data Analysis**

Học cái này trước các chứng chỉ Excel khác. Không đụng bảng tính thì bỏ cả phần Excel.

https://imp.i384100.net/5k0drD

**Google Data Analytics Professional Certificate**

Khóa rộng, hợp người mới.

https://imp.i384100.net/xJ6DQx

**Meta Data Analyst Professional Certificate**

Cùng kiểu với chứng chỉ Google. Lấy một cái là đủ, không cần sưu tầm bằng.

https://imp.i384100.net/rENKMG

**Google Advanced Data Analytics Professional Certificate**

Học sau chứng chỉ cơ bản. Học song song dễ bỏ giữa chừng.

https://imp.i384100.net/jR67bn

**Microsoft Power BI Data Analyst Professional Certificate**

https://imp.i384100.net/PzdQx6

**Work Smarter with Microsoft Excel**

Ngắn hơn chứng chỉ đầy đủ. Đủ nếu chỉ cần xong việc trên bảng tính.

https://imp.i384100.net/jR67ba

**Microsoft Excel Professional Certificate**

Khóa dài. Hợp khi bảng tính là việc hằng ngày.

https://imp.i384100.net/PzdQxq

**ChatGPT + Excel**

Học khi bạn đã tự làm được một bảng. Làm sớm hơn thì khó biết câu trả lời sai chỗ nào.

https://imp.i384100.net/rENKMB

Chọn nhánh nào thì đi nhánh đó. Đừng học nhánh kia cho có.

## Chặng 4 — Machine learning

Dành vài tuần cho một khóa.

Chỗ cần nắm là cách kiểm tra: máy đang học một quy luật, hay chỉ nhớ đáp án đã thấy. Học một khóa cho xong. Repo để tự gõ lại. Sách dày để tra khi kẹt, chưa cần đọc hết trước khi đi tiếp.

Thử giải thích câu này. Model đúng 99% trên dữ liệu nó đã thấy, sao vẫn có thể vô dụng? Nói được là qua chặng này.

### Machine Learning Specialization

Khóa của Andrew Ng. Chặng này chỉ học một khóa thì học khóa này đến cuối.

https://imp.i384100.net/k42K6M

### Machine Learning for Beginners

Repo miễn phí của Microsoft, có bài tập để gõ theo.

https://github.com/microsoft/ML-For-Beginners

### Made with ML

Đi từ notebook tới một thứ gần với sản phẩm hơn. Hợp khi video bắt đầu giống bài trên giấy, và bạn muốn làm một project.

https://github.com/GokuMohandas/Made-With-ML

### Understanding Deep Learning

Sách miễn phí của Simon J.D. Prince.

Mở bên cạnh khóa học lúc một chỗ bị khó. Không cần đọc xong sách rồi mới được học tiếp.

https://udlbook.github.io/udlbook/

### Designing Machine Learning Systems

Repo kèm sách của Chip Huyen.

Nói về hệ thống ML khi đã nằm trong sản phẩm. Để sau, chưa phải tuần đầu.

https://github.com/chiphuyen/dmls-book

## Chặng 5 — Deep learning

Chọn một khóa và học đến cuối. Hai khóa cùng lúc thường không xong khóa nào. Sách của Prince để cạnh những chỗ video đi nhanh.

### Deep Learning Specialization

Hợp nếu bạn muốn hiểu ý trước, rồi mới gõ code.

https://imp.i384100.net/qW6KMO

### Deep Learning with PyTorch, Keras and TensorFlow

Khóa của IBM. Hợp nếu bạn muốn dùng framework sớm.

https://imp.i384100.net/enZ0zg

Xong một khóa là sang chặng sau được. Không cần đăng ký cả hai.

## Chặng 6 — Làm việc với LLM

Đến đây video và sách dễ vào hơn, vì bạn đã biết training là gì. RAG với fine-tune cũng đỡ mơ hồ.

Làm sản phẩm thì thử lần lượt:

1. Viết prompt cho rõ
2. Thêm RAG, tức là cho model đọc tài liệu của bạn
3. Fine-tune, chỉ khi hai cách trên không đủ

Chỉ mua một cuốn thì mua *AI Engineering*.

### LLMs from Scratch

Video Stanford CS229, khoảng 1 giờ 44 phút.

Họ xây một LLM, không phải buổi demo sản phẩm. Xem sau khi bạn đã thấy một model được train. Ngày đầu chưa cần.

https://www.youtube.com/watch?v=9vM4p9NN0Ts

### AI Engineering

Sách của Chip Huyen, trả phí.

Cách làm sản phẩm quanh model có sẵn, không phải tự train model nền.

https://www.oreilly.com/library/view/ai-engineering/9781098166298/

### Prompt Engineering Guide

https://github.com/dair-ai/Prompt-Engineering-Guide

### LLM Course

Repo của Maxime Labonne.

Đi theo mục lục, từng phần. Mở hết tab một buổi rồi đóng lại thì không tính là đã học.

https://github.com/mlabonne/llm-course

### Hands-On Large Language Models

Code kèm sách của Jay Alammar và Maarten Grootendorst.

Cách dùng LLM trong ứng dụng. Không phải train một GPT từ đầu.

https://github.com/HandsOnLLM/Hands-On-Large-Language-Models

### Build a Large Language Model (From Scratch)

Sách của Sebastian Raschka, trả phí.

Tự code một model từ những phép tính nhỏ. Đọc khi bạn muốn hiểu bên trong. Tuần này cần ra sản phẩm thì để sau.

https://www.manning.com/books/build-a-large-language-model-from-scratch

### LLM Engineer's Handbook

Sách của Paul Iusztin và Maxime Labonne, trả phí.

Dữ liệu, fine-tune, đưa model lên máy chủ. Để sau, khi prompt và RAG không đủ cho việc bạn đang làm.

https://www.oreilly.com/library/view/llm-engineers-handbook/9781836200079/

## Chặng 7 — Bốn bài báo

Đọc chậm, nhưng vòng đầu chỉ đọc nông. Không cần viết bài tổng quan. Bốn bài này hay bị tutorial nhắc tên mà không nói chúng nói gì.

Với mỗi bài: đọc phần tóm tắt, xem hình, đọc mở đầu. Công thức để lại đến lúc đang làm và bị kẹt.

Kể lại được ReAct thành một vòng lặp, không cần mở bài ra, là đủ cho vòng này.

### 1. Chain-of-Thought Prompting

Wei và cộng sự, 2022.

Bảo model viết các bước ở giữa thì bài khó làm được hơn. Câu "nghĩ từng bước" bắt nguồn từ đây.

https://arxiv.org/abs/2201.11903

### 2. ReAct

Yao và cộng sự, 2022.

Nghĩ, gọi công cụ, xem kết quả, nghĩ tiếp. Phần lớn tutorial agent sau này đi theo khung này.

https://arxiv.org/abs/2210.03629

### 3. Toolformer

Schick và cộng sự, NeurIPS 2023.

Model có thể học lúc nào nên gọi API. Không cần làm lại nghiên cứu. Đọc để biết "dùng công cụ" không phải tính năng đặc biệt của một framework.

https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html

### 4. Generative Agents

Park và cộng sự, 2023.

Nhờ trí nhớ, một nhân vật giả còn nhớ chuyện hôm qua. Hữu ích khi agent của bạn quên sạch sau mỗi lượt chat.

https://arxiv.org/abs/2304.03442

## Chặng 8 — Xây agent

Đọc trước, xem sau, rồi mới xây. Bài viết ngắn hơn workshop, và nó nhắc một việc hay quên: nhiều lúc một câu lệnh với một công cụ là đủ, chưa cần dựng năm agent.

Làm từ repo có sẵn. Đừng mở file trống rồi đoán cấu trúc.

Có một agent gọi được một công cụ là được. Nó sẽ hỏng đôi lúc. Cố nói được vì sao.

### Đọc

**Building Effective Agents**

Anthropic. Phân biệt quy trình viết sẵn các bước với một agent tự quyết.

https://www.anthropic.com/engineering/building-effective-agents

**A Practical Guide to Building Agents**

PDF ngắn của OpenAI. Khi nào đáng làm agent, và làm xong thì trông nom thế nào. Hai bài này đủ thay một loạt video giới thiệu.

https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

**Agents**

Whitepaper của Google. Họ đặt tên các phần: model, công cụ, lớp điều phối. Đọc để nói cùng một thứ với người khác.

https://www.kaggle.com/whitepaper-agents

**Agents Companion**

Phần tiếp theo. Cách đánh giá, và những chỗ bài đầu chưa nói. Đọc khi bạn đã có agent nhưng chưa biết nó tốt hay chưa.

https://www.kaggle.com/whitepaper-agent-companion

**Claude Code best practices**

Thói quen khi để agent sửa code, để khỏi nhận một đống thay đổi mà mình không đọc. Chưa dùng agent để code thì bỏ qua.

https://code.claude.com/docs/en/best-practices

### Xem

**Agentic AI Overview**

Stanford, khoảng 57 phút. Từ một lần gọi model đến một hệ thống biết lập kế hoạch. Chưa phải buổi ngồi code.

https://www.youtube.com/watch?v=kJLiOGle3Lw

**Building and Evaluating Agents**

Sayash Kapoor, khoảng 20 phút. Demo chạy đẹp chưa có nghĩa sản phẩm đúng.

https://www.youtube.com/watch?v=d5EltXhbcfA

**Building Effective Agents**

Barry Zhang, Anthropic, khoảng 15 phút. Bản nói của bài viết cùng tên ở trên.

https://www.youtube.com/watch?v=D7_ipDqhtwk

**Building Agents with MCP**

Mahesh Murag, khoảng 1 giờ 44 phút. Cách gắn công cụ mà không phải viết lại mỗi lần đổi model. Xem sau khi bạn đã tự gọi được một công cụ.

https://www.youtube.com/watch?v=kQmXtrmQ5Zg

**Building an Agent from Scratch**

Kam Lasater, khoảng 19 phút. Gõ theo.

https://www.youtube.com/watch?v=xzXdLRUyjUg

**Philo Agents**

The Neural Maze, 6 video. Một project lớn: một làng các triết gia giả, có trí nhớ, RAG, và cách theo dõi hệ thống. Làm khi bạn muốn bài tập đụng nhiều tầng. Đừng bắt đầu từ đây.

Playlist:

https://www.youtube.com/playlist?list=PLacQJwuclt_sV-tfZmpT1Ov6jldHl30NR

Code của khóa:

https://github.com/neural-maze/philoagents-course

### Làm theo một repo

**AI Agents for Beginners**

Microsoft, bài xếp theo thứ tự.

https://github.com/microsoft/ai-agents-for-beginners

**GenAI Agents**

Nir Diamant. Mỗi notebook một kiểu agent. Bài gốc ghi repo này hai lần, nhưng chỉ có một repo.

https://github.com/NirDiamant/GenAI_Agents

**Hands-On AI Engineering**

Có agent, RAG, OCR. Chọn một project, chạy được, rồi sửa. Tải cả repo mà không chạy file nào thì chưa học được gì.

https://github.com/Sumanth077/Hands-On-AI-Engineering

**Awesome Generative AI Guide**

Mục lục thêm, khi bạn cần chủ đề file này không phủ. Không phải lịch học.

https://github.com/aishwaryanr/awesome-generative-ai-guide

### Sách, sau khi đã có một thứ chạy được

Ba cuốn dưới đều mất tiền. Chưa có agent nào chạy trên máy mình thì chưa cần mua.

**Building Applications with AI Agents**

Michael Albada. Cách thiết kế một agent, rồi đến nhiều agent.

https://www.oreilly.com/library/view/building-applications-with/9781098176495/

**AI Agents with MCP**

Kyle Stratis. Từ giao thức MCP đến server và client.

https://www.oreilly.com/library/view/ai-agents-with/9798341639546/

**AI Agents: The Definitive Guide**

Nicole Koenigstein. Đánh giá, an toàn, chi phí. Không phải cuốn để bắt đầu.

https://www.oreilly.com/library/view/ai-agents-the/0642572247775/

## Nên bỏ những gì

Người mới thường kẹt vì nhồi quá nhiều thứ vào một lịch, không phải vì thiếu link.

- Framework nhiều agent: để sau, khi một agent đã chạy.
- Fine-tune: để sau một prompt ổn và một bản RAG để so.
- Bốn chứng chỉ Excel: chỉ đáng nếu bảng tính là nghề của bạn.
- Đọc bài báo từ trang đầu đến trang cuối ngay vòng một: chưa cần.
- Thêm một video giới thiệu LLM nữa: video thứ hai lặp lại rất nhiều thứ video đầu đã nói.

## Lịch 6 tuần

Cho nhánh xây sản phẩm AI, khoảng 8 giờ mỗi tuần. Deep learning, sách trả phí và các chứng chỉ còn lại để vòng sau. Vòng sau bắt đầu khi bạn đã có một agent nhỏ, nó hỏng, và bạn biết nó hỏng vì đâu.

| Tuần | Việc |
| --- | --- |
| 1 | Xem video Karpathy. Bắt đầu Crash Course on Python. |
| 2 | Xong phần Python nền. Bắt đầu Machine Learning Specialization. |
| 3 | Học tiếp specialization. Đọc vài phần trong Prompt Engineering Guide. |
| 4 | Xong các khóa đầu của specialization. Đọc bài của Anthropic. Xem webinar Stanford về agent. |
| 5 | Làm bài trong repo Microsoft. Đọc ReAct, phần nông. Code theo video agent from scratch. |
| 6 | Làm một project trong GenAI Agents hoặc Hands-On AI Engineering. Ghi lại chỗ nó hỏng. |

## Tất cả link

Phần này để lấy link cho nhanh. Giải thích nằm ở các chặng phía trên.

### Video

- LLM Introduction — Karpathy: https://www.youtube.com/watch?v=zjkBMFhNj_g
- LLMs from Scratch — Stanford: https://www.youtube.com/watch?v=9vM4p9NN0Ts
- Agentic AI Overview — Stanford: https://www.youtube.com/watch?v=kJLiOGle3Lw
- Building and Evaluating Agents: https://www.youtube.com/watch?v=d5EltXhbcfA
- Building Effective Agents — Barry Zhang: https://www.youtube.com/watch?v=D7_ipDqhtwk
- Building Agents with MCP: https://www.youtube.com/watch?v=kQmXtrmQ5Zg
- Building an Agent from Scratch: https://www.youtube.com/watch?v=xzXdLRUyjUg
- Philo Agents: https://www.youtube.com/playlist?list=PLacQJwuclt_sV-tfZmpT1Ov6jldHl30NR

### GitHub

- GenAI Agents: https://github.com/NirDiamant/GenAI_Agents
- AI Agents for Beginners: https://github.com/microsoft/ai-agents-for-beginners
- Prompt Engineering Guide: https://github.com/dair-ai/Prompt-Engineering-Guide
- Hands-On Large Language Models: https://github.com/HandsOnLLM/Hands-On-Large-Language-Models
- Made with ML: https://github.com/GokuMohandas/Made-With-ML
- Hands-On AI Engineering: https://github.com/Sumanth077/Hands-On-AI-Engineering
- Awesome Generative AI Guide: https://github.com/aishwaryanr/awesome-generative-ai-guide
- Designing Machine Learning Systems: https://github.com/chiphuyen/dmls-book
- Machine Learning for Beginners: https://github.com/microsoft/ML-For-Beginners
- LLM Course: https://github.com/mlabonne/llm-course
- PhiloAgents, code: https://github.com/neural-maze/philoagents-course

### Bài hướng dẫn

- Agents, Google: https://www.kaggle.com/whitepaper-agents
- Agents Companion, Google: https://www.kaggle.com/whitepaper-agent-companion
- Building Effective Agents, Anthropic: https://www.anthropic.com/engineering/building-effective-agents
- Claude Code best practices: https://code.claude.com/docs/en/best-practices
- A Practical Guide to Building Agents, OpenAI: https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf

### Sách

- Understanding Deep Learning: https://udlbook.github.io/udlbook/
- Build a Large Language Model (From Scratch): https://www.manning.com/books/build-a-large-language-model-from-scratch
- LLM Engineer's Handbook: https://www.oreilly.com/library/view/llm-engineers-handbook/9781836200079/
- AI Engineering: https://www.oreilly.com/library/view/ai-engineering/9781098166298/
- Building Applications with AI Agents: https://www.oreilly.com/library/view/building-applications-with/9781098176495/
- AI Agents with MCP: https://www.oreilly.com/library/view/ai-agents-with/9798341639546/
- AI Agents: The Definitive Guide: https://www.oreilly.com/library/view/ai-agents-the/0642572247775/

### Bài báo

- Chain-of-Thought: https://arxiv.org/abs/2201.11903
- ReAct: https://arxiv.org/abs/2210.03629
- Toolformer: https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html
- Generative Agents: https://arxiv.org/abs/2304.03442

### Khóa học

- Crash Course on Python: https://imp.i384100.net/QYd9Dz
- Python for Data Science, AI & Development: https://imp.i384100.net/9Vd6kQ
- SQL Foundations: https://imp.i384100.net/k42K6v
- Introduction to Data Analytics: https://imp.i384100.net/5k0drn
- Excel Basics for Data Analysis: https://imp.i384100.net/5k0drD
- Google Data Analytics: https://imp.i384100.net/xJ6DQx
- Meta Data Analyst: https://imp.i384100.net/rENKMG
- Google Advanced Data Analytics: https://imp.i384100.net/jR67bn
- Microsoft Power BI: https://imp.i384100.net/PzdQx6
- Work Smarter with Microsoft Excel: https://imp.i384100.net/jR67ba
- Microsoft Excel Professional Certificate: https://imp.i384100.net/PzdQxq
- ChatGPT + Excel: https://imp.i384100.net/rENKMB
- Machine Learning Specialization: https://imp.i384100.net/k42K6M
- Deep Learning Specialization: https://imp.i384100.net/qW6KMO
- IBM Deep Learning with PyTorch, Keras and TensorFlow: https://imp.i384100.net/enZ0zg

---

Mình viết lại từ bộ tài liệu tổng hợp có sẵn. Link giữ nguyên. Thứ tự thì xếp lại cho người mới dễ đi.

Theo dõi [@Mohiniuni](https://x.com/Mohiniuni) để xem thêm về AI, làm đẹp và kinh doanh.
