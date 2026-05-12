# Lab 24 - Full Evaluation & Guardrail System 

---

Chào mừng đến với Lab 24! 

Đây là bài lab lớn nhất và quan trọng nhất của Chương 5. Sau 4 giờ thực hành, bạn sẽ xây dựng được một hệ thống đánh giá và bảo vệ (evaluation và guardrail system) sẵn sàng cho môi trường production cho RAG pipeline của mình , tương tự như cách các nhóm AI hàng đầu đang thực hiện vào năm 2026.

**Mục tiêu thật sự:** Giúp bạn có thể tự tin trả lời 3 câu hỏi cốt lõi về hệ thống AI:

1. 
**“Hệ thống này có hoạt động tốt không?"** (Evaluation) 


2. 
**"Khi người dùng tấn công, nó có chịu được không?"** (Guardrails) 


3. 
**"Khi nó hỏng, ta có biết kịp không?"** (Monitoring) 



---

Phần 1: Thông tin chung 

1.1 Tổng quan 

| Hạng mục | Chi tiết |
| --- | --- |
| **Tên lab** | Full Evaluation & Guardrail System 

 |
| **Thời gian** | 4 giờ (Phase A: 60' + B: 60' + C: 90' + Blueprint: 30') 

 |
| **Hình thức** | Cá nhân (khuyến khích) hoặc nhóm 2 người 

 |
| **Sản phẩm nộp** | GitHub repo + Blueprint document + Demo video 5 phút 

 |
| **Ngưỡng đạt** | 60/100 điểm 

 |
| **Xuất sắc** | <br>$\ge 90/100$ điểm 

 |

1.2 Bạn sẽ học được gì? 

Sau bài lab này, bạn sẽ có khả năng:

* 
**Triển khai:** RAGAS evaluation với 4 chỉ số cốt lõi (core metrics).


* 
**Xây dựng:** Pipeline LLM-as-Judge với so sánh cặp (pairwise comparison) và giảm thiểu định kiến (bias mitigation).


* 
**Phân tích:** Tính toán Cohen's kappa để hiểu mức độ đồng thuận (agreement scores) giữa người và máy.


* 
**Triển khai:** Các lớp bảo vệ đầu vào (PII + topic) và đầu ra (Llama Guard 3).


* 
**Đánh giá:** Đo lường độ trễ (latency overhead) và xác định các điểm nghẽn (bottleneck).


* 
**Sáng tạo:** Thiết kế tài liệu blueprint cho việc triển khai thực tế (production deployment).



---

Phần 2: Cấu trúc Lab 

Lab gồm 4 giai đoạn tuần tự. Bạn không nên bỏ qua bất kỳ giai đoạn nào vì mỗi phần sẽ xây dựng dựa trên kết quả của phần trước đó.

| Phase A (60') | Phase B (60') | Phase C (90') | Phase D (30') |
| --- | --- | --- | --- |
| **RAGAS Eval** | **LLM-as-Judge** | **Guardrails** | **Blueprint Document** |
| 30 điểm 

 | 25 điểm 

 | 35 điểm 

 | 10 điểm 

 |

2.1 Tips trước khi bắt đầu 

1. 
**Setup Git:** Khởi tạo repo và commit mỗi 30 phút để lưu lại lịch sử làm việc.


2. 
**Hiểu bức tranh lớn:** Đọc hết cả 4 phase trước khi bắt đầu viết code.


3. 
**Kiểm soát chi phí:** Lab này tiêu tốn khoảng $3-5$ USD; hãy theo dõi để không vượt quá $20$ USD.


4. 
**Hỗ trợ:** Nếu bị kẹt quá 20 phút, hãy đặt câu hỏi tại kênh Slack `#lab24-eval-guardrails`.



---

Phần 3 - Phase A: RAGAS Evaluation (60 phút, 30 điểm) 

**Mục tiêu:** Xây dựng pipeline đánh giá tự động cho hệ thống RAG từ Day 18.

Task A.1 - Synthetic Test Set Generation (15 phút) – 8 điểm 

Tạo bộ dữ liệu kiểm thử (test set) gồm 50 câu hỏi từ kho tài liệu với phân bổ:

* 
**50% Simple (single-hop):** Câu hỏi từ 1 đoạn văn (chunk).


* 
**25% Reasoning (multi-step):** Câu hỏi cần suy luận nhiều bước.


* 
**25% Multi-context:** Câu hỏi kết hợp thông tin từ từ 2 đoạn văn trở lên.



```python
# [cite_start]Code template mẫu cho Testset Generation [cite: 80, 81, 82, 85, 90, 94]
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# [cite_start]Load documents [cite: 86, 87]
loader = DirectoryLoader("./docs", glob="**/*.md")
documents = loader.load()

# [cite_start]Setup generator [cite: 89, 92]
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings()
)

# [cite_start]Generate test set [cite: 93, 96, 97]
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=50,
    distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}
)
[cite_start]testset.to_pandas().to_csv("testset_v1.csv", index=False) [cite: 104]

```

**Tiêu chí nghiệm thu:**

* File `testset_v1.csv` có ít nhất 50 dòng và đủ 4 cột: `question`, `ground_truth`, `contexts`, `evolution_type`.


* Thực hiện review thủ công ít nhất 10 câu hỏi và ghi chú vào `testset_review_notes.md`.



---

Phần 4 - Phase B: LLM-as-Judge & Calibration (60 phút, 25 điểm) 

**Mục tiêu:** Xây dựng pipeline giám định bằng LLM để đo lường các yếu tố mà RAGAS chưa bao phủ.

Task B.1 - Pairwise Judge Pipeline (20 phút) – 10 điểm 

Xây dựng Judge để so sánh hai phiên bản câu trả lời khác nhau (ví dụ: có và không có re-ranker). Sử dụng kỹ thuật **Swap-and-average** để giảm thiểu định kiến về vị trí (position bias).

Task B.3 - Human Calibration với Cohen's Kappa (20 phút) – 8 điểm 

Thực hiện dán nhãn thủ công (human-label) cho 10 cặp câu hỏi và tính toán chỉ số Cohen's Kappa để so sánh với kết quả của Judge.

**Bảng diễn giải chỉ số Kappa:** 

* 
**Kappa < 0:** Tệ hơn ngẫu nhiên (Judge sai hệ thống).


* 
**0 - 0.2:** Đồng thuận rất thấp (Không tin tưởng được).


* 
**0.4 - 0.6:** Đồng thuận mức trung bình (Có thể dùng để giám sát).


* 
**$\ge 0.6$:** Sẵn sàng cho môi trường Production.



---

Phần 5 - Phase C: Guardrails Stack (90 phút, 35 điểm) 

**Mục tiêu:** Xây dựng hệ thống bảo vệ đa lớp (defense-in-depth) để ngăn chặn lỗi tiếp cận người dùng.

Task C.1 - Input Guardrail: PII Redaction (20 phút) – 8 điểm 

Sử dụng thư viện Presidio kết hợp với các biểu thức chính quy (regex) tùy chỉnh cho dữ liệu Việt Nam (như CCCD, số điện thoại VN) để ẩn danh thông tin nhạy cảm.

Task C.4 - Output Guardrail: Llama Guard 3 (20 phút) – 8 điểm 

Triển khai Llama Guard 3 để kiểm tra độ an toàn của câu trả lời trước khi gửi tới người dùng. Nếu không có GPU, bạn có thể sử dụng Groq API (miễn phí) để thực hiện tác vụ này.

---

Phần 6 - Phase D: Blueprint Document (30 phút, 10 điểm) 

**Mục tiêu:** Tổng hợp toàn bộ quá trình làm việc thành một tài liệu Blueprint dài 4-6 trang.

Section 1: SLO Definition 

Xác định ít nhất 5 chỉ số mức độ dịch vụ (SLO) với các ngưỡng cảnh báo cụ thể:

* 
**Faithfulness:** Mục tiêu $\ge 0.85$, Cảnh báo nếu $< 0.80$ trong 30 phút.


* 
**P95 Latency:** Mục tiêu $< 2.5s$, Cảnh báo nếu $> 3s$ trong 5 phút.



---

Phần 7 - Submission 

**Cấu trúc Repo bắt buộc:** 

* 
`README.md`: Tổng quan dự án (200-300 từ).


* 
`phase-a/`: Chứa file `testset_v1.csv` và `failure_analysis.md`.


* 
`phase-b/`: Chứa `pairwise_results.csv` và `judge_bias_report.md`.


* 
`phase-c/`: Các file script `input_guard.py`, `output_guard.py`.


* 
`phase-d/`: Tài liệu `blueprint.md`.


* 
`demo/`: Video demo dài 5 phút.



---

Phần 10 - FAQ 

* 
**Q1: Tôi không có GPU, làm sao chạy Llama Guard?** 


* 
**A:** Sử dụng Groq API (miễn phí) để suy luận mẫu Llama Guard 3.




* 
**Q3: Cohen's kappa = -0.1, tôi làm sai ở đâu?** 


* **A:** Có thể bạn đang dán nhãn dựa trên vị trí cột thay vì nội dung câu trả lời. Hãy kiểm tra lại tính nhất quán của dữ liệu.




* 
**Q7: Tôi có thể nộp muộn không?** 


* **A:** Chính sách mặc định: trừ 10% điểm cho mỗi ngày nộp muộn (tối đa 3 ngày). Sau đó sẽ nhận 0 điểm.
