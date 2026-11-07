# Đánh giá WAGUR1 cho mục tiêu công bố quốc tế

**Ngày đánh giá:** 07/11/2026, Asia/Bangkok.  
**Repo:** `D:\STUDY\LAB\TEST\WAGUR1` — tên dự án trong mã là `scope_vcm`, không phải một mô hình tên WAGUR1.  
**Commit được đọc:** `032d3faea8665dedd681251e0626cccc1cfff146`.  
**Loại đánh giá:** bản đánh giá kỹ thuật và nghiên cứu trước khi nộp bài, theo yêu cầu người dùng; có hỗ trợ AI. Không phải phản biện được một tạp chí chỉ định hoặc quyết định của ban biên tập.

## 1. Kết luận và điểm số

**Điểm tổng thể hiện tại: 4,5/10 cho hồ sơ có thể dùng để nộp một bài nghiên cứu quốc tế.** Repo có câu hỏi nghiên cứu rõ, có ý thức kiểm soát baseline và kiểm tra khả năng bác bỏ giả thuyết. Tuy nhiên, ở snapshot này, pipeline báo cáo có lỗi tái hiện được; baseline quyết định A1 chưa được tích hợp thành đường chạy đã xác minh; A3 chưa có kết quả kiểm chứng trong repo; các con số trong nhật ký chưa kèm gói dữ liệu để đối soát.

**Khuyến nghị hiện tại: chưa nên dùng snapshot này làm bộ bằng chứng cho một bài nghiên cứu hoàn chỉnh.** Nên hoàn thiện E1 và đóng gói kết quả trước. Hướng bài đánh giá thực nghiệm về pre-editing, ROI và cách diễn giải rate–accuracy có triển vọng hơn việc trình bày repo như một phương pháp video mới đã được chứng minh.

Điểm này đánh giá **mức hỗ trợ của tài liệu, mã và bằng chứng đang có**, không đánh giá năng lực tác giả, không phải xác suất được nhận, và không quy đổi sang Q1/Q2/Q3. Chưa có bản thảo hoặc tạp chí đích nên không thể chấm mức phù hợp với một nơi công bố cụ thể. Thang điểm dưới đây là rubric tư vấn của lần đánh giá này, chưa được hiệu chuẩn giữa nhiều người chấm.

| Tiêu chí | Tối đa | Điểm hiện tại | Căn cứ chính |
|---|---:|---:|---|
| Câu hỏi và thiết kế nghiên cứu | 10 | 8 | Có A0–A4, control blur-all, gate và phương án khi giả thuyết không được hỗ trợ |
| Tính mới và định vị đóng góp | 15 | 7 | So sánh matched-mask và kiểm toán accuracy đáng làm; blur và pre-editing đã có tiền lệ |
| Toán học và cách đo | 15 | 8 | Có PCHIP, overlap, ceiling, absolute-mAP; còn vấn đề estimand và suy diễn F2 |
| Tính đúng và độ hoàn thiện của mã | 15 | 7 | Module tương đối gọn; có lỗi report, ROI x264, A4 và co-primary |
| Bằng chứng thực nghiệm có thể kiểm chứng | 20 | 5 | Có nhật ký chạy thật, nhưng không có CSV/manifest/checkpoint kết quả trong bản checkout |
| Thống kê và độ vững kết luận | 10 | 5 | Có paired test, bootstrap; thiếu power rationale đúng, xử lý nhiều phép thử và protocol nhất quán |
| Khả năng tái lập độc lập | 5 | 2 | Có hướng dẫn và config; thiếu environment lock, hash đầu vào và archive chạy |
| Hồ sơ bài báo và tài liệu tham khảo | 10 | 3 | Có tài liệu phương pháp; chưa thấy manuscript, bảng/hình kết quả hoàn chỉnh hoặc thư mục tham khảo chuẩn |
| **Tổng** | **100** | **45** | **4,5/10** |

## 2. Phạm vi và cách kiểm chứng

Đã đọc README, MATH, PROTOCOL, COUNCIL, SELF_REVIEW, SESSION_LOG, config, các module xử lý codec/pre-edit/mask/metric, script encode/eval/report, test BD và phần triển khai Kaggle liên quan. Không chạy các lệnh đẩy job Kaggle, không đọc token hoặc thông tin đăng nhập. Không sửa mã thuật toán của repo trong lần đánh giá này.

Các mức bằng chứng được phân biệt như sau:

- **Đã chạy kiểm chứng:** unit test và các ví dụ dữ liệu tổng hợp nhỏ trên máy hiện tại.
- **Quan sát trực tiếp từ mã:** cấu trúc lệnh encoder, schema, logic resume, metadata và adapter.
- **Repo tự báo cáo:** các kết quả Kaggle trong SESSION_LOG; chưa tái tính từ đầu ra gốc.
- **Đối chiếu công khai:** chỉ tra cứu tài liệu và công trình đã công khai; không gửi nội dung riêng hoặc thông tin tài khoản của repo ra công cụ tìm kiếm.

Không thấy dữ liệu video, checkpoint, CSV kết quả, manifest chạy hoặc bản thảo PDF trong cây file được khảo sát. Điều này **không chứng minh các thí nghiệm chưa từng chạy**; nó có nghĩa người đánh giá chưa thể kiểm chứng chúng từ repo này. Không có FFmpeg trên PATH của môi trường kiểm tra, nên chưa chạy lại encode/decode, scorer hoặc SCOPE với dữ liệu thật.

## 3. Repo thực sự đang làm gì?

E1 hỏi liệu sửa ảnh trước nén có tốt hơn phân bổ bit bằng ROI delta-QP trong codec hay không, trong điều kiện cùng nguồn mask và cùng task model. Các nhánh gồm:

| Nhánh | Vai trò | Trạng thái quan sát được |
|---|---|---|
| A0 | Nén ảnh gốc | Có wrapper x264/x265 |
| A1 | ROI delta-QP, không sửa pixel nguồn | x265 cần đường map chưa được xác minh/tích hợp; x264 hiện bỏ qua ROI trong wrapper chính |
| A2 | Giữ ROI, làm mờ nền, không huấn luyện | Có mã; nhật ký ghi thí nghiệm thật |
| A2c | Làm mờ toàn ảnh | Có control; không bật trong config mẫu |
| A3 | Chạy SCOPE học sẵn từng frame | Có adapter và tải weight ở kernel; không có đầu ra A3 đã xác minh trong checkout |
| A4 | Pre-edit kết hợp ROI | Có encode path; eval thiếu ceiling cần cho report |
| E2/E2t | Tương tác với công cụ neural và thời gian | Chủ yếu là kế hoạch; không coi là đóng góp đã hoàn thành |

Việc áp dụng mô hình ảnh từng frame chưa tạo thành một phương pháp thời gian mới. Đóng góp hiện tại nằm chủ yếu ở **thiết kế đánh giá và bộ khung thí nghiệm**.

## 4. Ưu điểm

### 4.1. Chọn đúng phép so sánh có thể làm thay đổi kết luận

Đặt A1 làm baseline chính là một lựa chọn tốt: nếu thay đổi phân bổ QP đã đạt mục tiêu, tác giả cần giải thích vì sao còn phải sửa nội dung ảnh. Dùng chung mask cho A1/A2 giúp giảm một yếu tố gây nhiễu. Tuy nhiên, ưu điểm này mới là thiết kế; phải thực hiện được A1 mới trở thành bằng chứng.

### 4.2. Có ý thức tách giảm bitrate khỏi mất thông tin task

Đo chất lượng trước nén của ảnh gốc và ảnh đã sửa, báo cáo eta, đồng thời có absolute-mAP là hữu ích. Đây là cách phát hiện trường hợp bitrate giảm nhưng chất lượng downstream đã mất từ bước tiền xử lý. Control blur-all cũng giúp khảo sát vai trò của tính chọn lọc.

### 4.3. Có nguyên tắc kiểm soát việc lựa chọn kết quả

MATH/PROTOCOL nêu gate, trường hợp BD không xác định, kết quả theo sequence và phạm vi bác bỏ A2 so với A3. Tinh thần này tốt hơn việc chỉ đưa trung bình đẹp. Cần bảo đảm implementation và lịch sử sửa protocol thực sự tuân theo những nguyên tắc đó.

### 4.4. Có nền tảng phần mềm để tiếp tục nghiên cứu

Codec, pre-edit, mask, thống kê và reporting được tách module. Có cơ chế lưu kết quả, cảnh báo option encoder không hỗ trợ và đọc mask từng frame ở encode pass. **13 unit test hiện có đều chạy đạt** trong lần kiểm tra này. Những test đó xác nhận một số tính chất cơ bản của BD và gate; chưa xác nhận toàn pipeline.

### 4.5. Có ghi nhận kết quả bất lợi và lỗi vận hành

SESSION_LOG ghi ceiling drop lớn, khó khăn ROI và các lần thử không thành công. SELF_REVIEW liệt kê lỗi từng gặp. Đây là dữ liệu phát triển hữu ích, dù nhật ký tự đánh giá không thay thế dữ liệu thực nghiệm hoặc phản biện độc lập.

## 5. Các lỗi và khoảng trống kỹ thuật quan trọng

**P1** dưới đây là vấn đề có thể làm báo cáo hỏng hoặc làm sai quyết định nghiên cứu. **P2** là vấn đề ảnh hưởng tính hợp lệ, độ vững hoặc khả năng tái lập. Đây là phân loại ưu tiên sửa, không phải nhận định về tác giả.

### P1-01 — Report không đọc được đúng schema do runner tạo ra

**Vị trí:** `scope_vcm/experiment.py:30`; `scripts/report_e1.py:41`, `:130`.

Runner ghi `quality` trống vào `e1.csv`; eval ghi `quality` vào `quality.csv`. Khi merge, pandas tạo `quality_x` và `quality_y`. Report sau đó đọc cột `quality`, gây **`KeyError: 'quality'`**. Đã tái hiện bằng fixture theo đúng schema runner, không cần video.

**Tác động:** đường encode → eval → report được hướng dẫn trong README không thể được coi là đã xác minh chỉ nhờ test BD. **Cần sửa:** thống nhất schema; lấy quality từ eval một cách tường minh; thêm một kiểm tra tích hợp dùng schema thật.

### P1-02 — Hai nhánh co-primary không được xét theo quy tắc đã mô tả

**Vị trí:** `scripts/report_e1.py:83–89`, `:183–200`; `docs/MATH.md`, mục 6.

Khi có A3, report chọn A3 để tính BD/gate cuối cùng. A2 chỉ còn được liệt kê ở phần ceiling. Fixture có A2 tiết kiệm rate hơn ROI và vượt gate, trong khi A3 thất bại, vẫn nhận **NO-GO**. Với fixture đó, A2 có gap −20 điểm phần trăm và p = 0,00390625.

**Cần sửa:** tính kết quả riêng cho từng nhánh, kết hợp đúng quy tắc đã đăng ký, đồng thời xử lý multiplicity. Không suy ra cả giả thuyết chết chỉ vì nhánh được ưu tiên trong mã thất bại.

### P1-03 — Thiếu baseline vẫn bị diễn giải thành NO-GO khoa học

**Vị trí:** `scripts/report_e1.py:153`, `:199–200`; `scope_vcm/metrics.py:22`; `kaggle/real_e1_kernel.py:153`.

Nếu thiếu A1, gap là NaN và gate không đạt; report vẫn in thông điệp pivot và “publish E1 as null + ROI baseline”. Fixture không có A1 đã tái hiện đúng hành vi này. Đây là **thí nghiệm chưa đủ điều kiện kết luận**, không phải bằng chứng pre-editing thua ROI. Tương tự, không đủ số sequence hoặc đường cong không có overlap cần được giải thích đúng phạm vi.

**Cần sửa:** phân biệt `INCOMPLETE/INVALID`, `INCONCLUSIVE`, và kết quả hợp lệ hỗ trợ/không hỗ trợ giả thuyết. Không có ý nghĩa thống kê cũng không tự chứng minh tương đương hoặc vô hiệu.

### P1-04 — x264 bỏ qua yêu cầu ROI trong wrapper chính

**Vị trí:** `scope_vcm/codec.py:139–143`.

Nhánh x264 không đọc `roi_map`/`roi_delta`. Kiểm tra bằng mock cho thấy lệnh encode A0 và encode có ROI giống nhau, ngoại trừ đường dẫn đầu ra. Vì vậy, nếu bật A1 với x264 qua runner hiện tại, có thể tạo một nhánh mang tên ROI nhưng không có can thiệp ROI.

Đối với x265, mã đã thừa nhận `qp-delta-map` không phải option public thông thường và thiết kế cho thất bại rõ khi option không được hỗ trợ. **Cần sửa:** từ chối ROI x264 khi chưa hỗ trợ; tích hợp một backend ROI thật; lưu bằng chứng sign/granularity và kiểm tra offset bằng 0 khớp anchor. Các script ROI hunt chưa thay thế được integration đó.

### P1-05 — A4 không có ceiling để report chuẩn hóa

**Vị trí:** `scripts/eval_e1.py:83–84`; `scripts/report_e1.py:32`, `:127–128`; config mẫu bật A4.

Eval tạo ceiling cho A2/A2c/A3, không tạo hoặc ánh xạ ceiling cho A4. Report lại coi A4 là content arm và tìm ceiling A4. Đã tái hiện **`KeyError: ('s0', 'A4_preedit_fb_roi')`**. Nếu A4 dùng chính ảnh A2, có thể kế thừa ceiling A2 bằng quy tắc rõ ràng; không nhất thiết encode ceiling lần nữa.

### P1-06 — Lưới QP chính chưa được thực thi thống nhất

**Vị trí:** `docs/MATH.md:119`; `docs/SESSION_LOG.md:65`; `docs/PROTOCOL.md:14`; `configs/e1_decisive.yaml`; `scripts/report_e1.py`.

MATH và SESSION_LOG nêu 5 mức QP chính, 4 mức trung gian chỉ để sensitivity. PROTOCOL/config vẫn đưa 9 mức; report không lọc QP theo tập chính, và `--config` chỉ phục vụ domain. Dữ liệu 9 mức sẽ đi vào BD với cả 9 điểm.

**Cần sửa:** khóa `primary_qps` và `sensitivity_qps`; xuất hai bảng riêng; ghi version/amendment cùng thời điểm sửa. Nhật ký cho biết đã có kết quả một số run trước quyết định đổi lưới, nên không thể dùng nhãn “pre-registered” chung cho mọi phân tích mà không giải trình trình tự. Đây là vấn đề truy vết, chưa phải bằng chứng chọn kết quả có chủ ý.

### P2-01 — Kiểm tra join chưa bảo đảm một hàng tương ứng một hàng

**Vị trí:** `scripts/report_e1.py:41–46`.

Chỉ so độ dài sau inner join không phát hiện đầy đủ khóa trùng hoặc thiếu. Fixture bỏ một quality row và nhân đôi một row khác vẫn được chấp nhận: 120 hàng đầu vào và 120 hàng sau merge. **Cần sửa:** khóa duy nhất `(seq, arm, qp, delta)`, merge `validate='one_to_one'`, kiểm tra đủ keys và số hữu hạn. Đây là lỗi độc lập với P1-01.

### P2-02 — Hàm làm sạch BD có thể tạo lại quality trùng

**Vị trí:** `scope_vcm/bd.py:53–58`.

Với điểm `(quality, rate) = [(0.1,100), (0.2,200), (0.3,100), (0.4,300)]`, bước gộp rate tạo `[(0.2,100), (0.2,200), (0.4,300)]`; PCHIP báo **`ValueError: x must be strictly increasing sequence`**, ngay cả khi so đường cong với chính nó.

**Cần sửa:** tách preprocessing của BD-rate và BD-quality theo trục nội suy; không thay đổi trục còn lại một cách ngầm định. Cần công khai cách xử lý rate–accuracy không đơn điệu, dominance và overlap. PCHIP không tự biến một đường cong không đơn điệu thành đường cong đơn điệu.

### P2-03 — Các tham số video và nguồn dữ liệu chưa khóa đầy đủ

**Vị trí:** `scope_vcm/codec.py:105`, `:139–148`; `scope_vcm/experiment.py:50`, `:119`; `kaggle/real_e1_kernel.py:42–46`, `:149`.

Protocol gọi low-delay nhưng wrapper chưa khóa GOP/B-frame/lookahead để bảo đảm điều đó. Tài liệu x265 xác định `bframes=0` để dùng P/I với độ trễ thấp; CQP đơn thuần không đủ. Cần công bố cấu hình và bitstream thực tế, tránh gọi wrapper hiện tại là phép tái lập CTC đầy đủ. [Tài liệu x265 CLI](https://x265.readthedocs.io/en/master/cli.html#cmdoption-bframes)

Kernel mặc định lấy 150 frame, resize 960×540, 3 QP và 7 sequence. Có wrapper cho phép đổi frame/QP, nên phải lấy manifest từng run để biết đã chạy gì. Runner chính đọc frame từ `0000.png`; dữ liệu MOT gốc cần bước chuyển đổi tên/định dạng. FPS đang dùng chung cấu hình, chưa lấy metadata riêng từng sequence. Không nên diễn giải clip ngắn, resize thành kết luận cho video nguyên độ phân giải và toàn thời lượng.

### P2-04 — Resume và manifest chưa đủ chống trộn thí nghiệm

**Vị trí:** `scope_vcm/experiment.py:74–93`, `:126`, `:159`; `scripts/make_masks.py:48`; `scripts/eval_e1.py:68–70`.

Cache dựa trên khóa sequence/arm/QP/delta, sự tồn tại hoặc số frame. Đổi blur sigma, ảnh nguồn, mask, weight hay scorer nhưng dùng lại workdir có thể tái sử dụng dữ liệu cũ. Config hash mới có được ghi, nhưng không được dùng để bác bỏ cache không tương thích.

Manifest hiện có git SHA và một số cấu hình, chưa thấy encoder version strings, checkpoint hash, mask hash, dữ liệu đầu vào hoặc environment đầy đủ như tài liệu hứa. Kernel copy CSV rồi xóa workdir, không đóng gói thư mục `results/manifests` vào `out/`. **Cần sửa:** run ID bất biến, xác thực cache, khóa môi trường, bảo toàn manifest và command/log cần thiết. Có đường dẫn weight không tương đương với việc đã pin nội dung weight.

### P2-05 — Các lỗi phụ làm suy yếu độ tin cậy của smoke test

- `scripts/check_encoders.py:50–56` giải mã PNG mà không đặt `-start_number 0`, rồi tìm `0000.png`. Cần đồng bộ chỉ số và tách lỗi đánh số khỏi sai khác chuyển màu RGB↔YUV; lossless ở miền codec không tự bảo đảm RGB round-trip tuyệt đối. Chưa chạy lại kiểm tra native này do thiếu FFmpeg.
- `kaggle/kernel_driver.py:101–105` sử dụng `cfg` trước khi biến này được gán ở dòng 119. Nếu tìm được checkpoint, nhánh đó có thể bị exception handler biến thành “SCOPE failed”, khiến A3 không bật. Đây là quan sát từ mã, chưa chạy tải weight.
- `kaggle/roi_hunt_x265.c:70` truyền địa chỉ một `x265_picture` trên stack vào hàm free; cả harness x264/x265 chỉ flush một lần. Cần sửa vòng đời bộ nhớ và drain encoder đến hết trước khi dùng kích thước bitstream để kết luận ROI. Chưa biên dịch lại trong lần đánh giá này. [Tài liệu x265 API](https://x265.readthedocs.io/en/master/api.html)
- Encode pass đọc mask từng frame, nhưng `YOLOMaskProvider.__call__` vẫn tích lũy mask toàn sequence trong dict. Claim streaming không áp dụng cho toàn bộ pipeline; cần tính bộ nhớ khi mở rộng dữ liệu.

## 6. Những vấn đề nghiên cứu cần sửa cách lập luận

### 6.1. Chuẩn hóa theo ceiling trả lời một câu hỏi khác

MATH nói raw-mAP BD trở nên không hợp lệ chỉ vì sửa nội dung. Phát biểu này quá mạnh. Nếu mọi ảnh sau giải mã đều được chấm trên cùng ground truth và cùng metric, bitrate tại cùng **absolute task accuracy** vẫn là đối tượng so sánh có nghĩa. Pre-editing có thể được xem là một thành phần của hệ thống nén.

Chuẩn hóa `m/m*` hữu ích để nghiên cứu giữ lại bao nhiêu phần chất lượng của từng nhánh, nhưng không bảo đảm công bằng cho mục tiêu deployment. Ví dụ ceiling gốc 0,50, ceiling edited 0,25: cùng normalized accuracy 0,80 tương ứng absolute mAP 0,40 và 0,20. Hai mức này không đáp ứng cùng yêu cầu.

**Đề nghị:** lấy absolute rate–accuracy làm trục chính cho claim triển khai; normalized metric và eta dùng để phân tích mất thông tin và độ bền dưới nén. Nếu vẫn giữ normalized metric làm chính, cần nói rõ estimand và tránh mô tả nó như cách sửa duy nhất của BD. Nên vẽ đường cong, công bố overlap và kiểm tra độ nhạy nội suy vì gate chỉ 1 điểm phần trăm. [The Bjøntegaard Bible](https://arxiv.org/abs/2304.12852)

### 6.2. F2 là quy tắc quyết định, chưa phải bằng chứng nhân quả đầy đủ

Eta vượt 1% chứng minh nhánh không đáp ứng ngưỡng chất lượng mà protocol đặt ra. Nó **không tự chứng minh toàn bộ tiết kiệm rate đến từ “content destruction”**. Để tách cơ chế, cần absolute-mAP curves, control, ví dụ lỗi detector và phân tích ở các mức bitrate/chất lượng tương ứng.

Tương tự, F2 trên A2 làm mờ nền không bác bỏ mọi phương pháp learned pre-editing hoặc A3. Các ngưỡng eta, 1% và 3% cần có lý do thực dụng và sensitivity; không phải hằng số khoa học chung.

### 6.3. “Bounded” và “lossless ROI” cần dùng chính xác

Giới hạn alpha/beta chỉ ràng buộc residual branch. Thành phần blur có thể làm thay đổi pixel lớn, kể cả khi alpha = beta = 0. Vì vậy không suy ra tổng perturbation nhỏ hoặc người xem chấp nhận được chỉ từ hai bound. Background fidelity floor được nhắc trong MATH nhưng chưa thấy đầu ra PSNR tương ứng ở pipeline đã đọc.

ROI delta-QP giữ nguyên **pixel đầu vào encoder**, nhưng nén với QP hữu hạn vẫn mất mát. Nên dùng “ROI allocation without source pre-editing” hoặc giải thích rõ thuật ngữ; tránh khiến người đọc hiểu baseline nén lossless.

### 6.4. Matched mask chưa được bảo đảm cho A3

A2 nhận mask cache chung. `ScopePreEditor` gọi model upstream, không truyền mask cache đó vào API; không có bằng chứng attention của A3 trùng mask A1. Nếu A3 dùng attention riêng, phép so sánh khác cả editor lẫn mask. Cần xuất attention A3 để làm baseline tương ứng, hoặc báo cáo riêng ablation mask và hệ thống hoàn chỉnh. Adapter tồn tại chưa chứng minh điều kiện công bằng đã đạt.

### 6.5. Thống kê cần thận trọng hơn

- **7 sequence không đạt quy tắc n≥8 mà repo tự đặt.** Các bản lặp trên GPU/tài khoản khác chủ yếu kiểm tra độ lặp lại; 7 video chạy ba lần không trở thành 21 sequence độc lập.
- MATH giải thích n≥8 vì ít hơn thì Wilcoxon exact không thể đạt p<0,05. Lý do đó sai: kiểm tra thực tế với 5 chênh lệch âm, không tie, cho **p=0,03125**. Vẫn có thể giữ n≥8 theo protocol, nhưng cần lý do độ chính xác/power hợp lý hơn. [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
- Wilcoxon signed-rank có giả định về phân phối chênh lệch; sign test là kiểm tra độ vững hữu ích. Bootstrap theo sequence với n nhỏ không tự bảo đảm CI ổn định. Cần báo cáo đơn vị suy luận, sequence selection và mức bất định.
- Nếu chỉ cần một trong A2/A3 vượt kiểm định để tuyên bố thành công, cần quy tắc kiểm soát nhiều phép thử, ví dụ Holm hoặc quy trình phân cấp xác định trước.
- Code kiểm định chênh lệch so với 0, kèm điều kiện median≤−1. Đó không tương đương kiểm định hiệu ứng thật vượt biên −1 với độ tin cậy 95%. Cần nói đúng ý nghĩa gate hoặc xây dựng kiểm định theo margin.
- `BD(A0,A2) − BD(A0,A1)` là **chênh lệch điểm phần trăm BD so với A0**, không trực tiếp là phần trăm tiết kiệm rate so với A1. Hai BD còn có thể dùng overlap khác nhau. Nên báo cáo thêm BD trực tiếp A1→A2/A3 trên miền chất lượng chung phù hợp.
- Chọn delta tốt nhất trên tập đánh giá là baseline oracle khá khắt khe với pre-editing; cần gọi đúng tên, báo cáo chi phí tìm delta và bổ sung lựa chọn delta bằng validation nếu muốn claim triển khai thực tế.

### 6.6. Ground truth, category mapping và chi phí hệ thống

`scope_vcm/detect.py:93` dùng `class_id + 1` cho YOLO. Cách này đúng với person ở MOT nhưng **không đúng cho toàn bộ COCO 80→91** vì category ID không liên tục. Ví dụ chỉ số 11 của mô hình phải ánh xạ sang COCO ID 13, không phải 12. Cần mapping chuẩn trước khi mở rộng sang COCO đa lớp. [Mã converter chính thức Ultralytics](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/data/converter.py)

`scripts/mot_to_coco.py` lọc visibility và box nhỏ; đây là tập con được định nghĩa riêng, không tự tương đương benchmark MOT/VCM chuẩn. Cần giải thích ignore regions và cách tránh phạt detection đúng trên đối tượng bị loại. `mAP` hiện giao img_ids với GT mà không bắt buộc khớp toàn bộ, có thể thu nhỏ tập đánh giá ngoài ý muốn.

Mask detector, pre-editor và scorer cần tách vai trò; có ít nhất một scorer độc lập với model sinh mask. Ngoài bitrate, cần đo chi phí tạo mask, pre-edit, encode/decode, bộ nhớ và latency. Protocol hứa decode time nhưng schema hiện chưa ghi. Chưa có tracking pipeline được xác minh để dùng IDF1/MOTA làm kết quả đã hoàn thành.

## 7. Tính mới và bằng chứng kết quả hiện có

### 7.1. Những gì chưa thể coi là mới

Pre-edit ảnh để tối ưu bitrate–detection đã được SCOPE công bố; repo WAGUR1 dùng lại mô hình đó qua adapter. Tiền xử lý video cho machine vision với codec chuẩn cũng đã có công trình của Zhao và cộng sự. Vì thế “đưa SCOPE sang từng frame video” hoặc “blur background để giảm bit” chưa đủ làm đóng góp chính. [SCOPE](https://arxiv.org/abs/2610.39778), [Zhao và cộng sự](https://arxiv.org/abs/2512.15331)

Hướng đóng góp đáng phát triển là: đối chiếu các editor với ROI dưới điều kiện kiểm soát rõ; chứng minh trường hợp metric tương đối và absolute accuracy cho kết luận khác nhau; mô tả miền hiệu quả/thất bại qua dữ liệu có thể tái lập. Đây là nhận định định hướng, chưa phải khẳng định novelty đã được xác lập. Các tuyên bố “chưa ai so sánh”, “baseline chưa ai vượt” trong tài liệu cần literature matrix và nguồn trực tiếp, đặc biệt tài liệu MPEG m64712 chưa được xác minh độc lập trong lần đánh giá này.

### 7.2. Các số liệu trong SESSION_LOG chỉ được ghi nhận là tự báo cáo

| Nội dung nhật ký | Có thể nói ở thời điểm này |
|---|---|
| 19 shard hoàn tất, GPU T4 | Repo ghi nhận như vậy; chưa có log/manifest gốc để đối chiếu |
| FRCNN: F2 7/7, eta khoảng 0,028–0,516, median BD khoảng −62,1% | Tín hiệu đáng khảo sát; chưa tái tính từ CSV |
| Các run khác: median khoảng −67,3%, F2 7/7 | Cần tách scorer, frame limit và lưới QP; một dòng log vừa gọi 5 CTC vừa liệt kê 3 QP |
| Không lệch bitrate giữa replicates | Chưa thấy bảng ghép cặp/hash; 19 shard cũng không đủ ba bản lặp cho cả bảy sequence |
| “Scorer-robust”, “content-destruction finding” | Cần nguồn dữ liệu, protocol và phân tích cơ chế tương ứng trước khi dùng như kết luận bài báo |

Lỗi report ở snapshot hiện tại không chứng minh mọi kết quả cũ sai: có thể chúng được tính bằng script khác hoặc dữ liệu đã biến đổi. Cần cung cấp **script và commit thực sự sinh từng bảng**, thay vì chỉ suy luận từ nhật ký.

## 8. Lộ trình ưu tiên trước khi nộp

| Ưu tiên | Việc cần hoàn thành | Tiêu chí nghiệm thu |
|---|---|---|
| 1 | Sửa schema và logic report | Fixture schema thật chạy; thiếu A1/n/quality nhận trạng thái thích hợp; A2/A3 và A4 được xử lý đúng |
| 2 | Làm baseline ROI hoạt động | Offset=0 khớp anchor; kiểm chứng dấu và vùng ảnh hưởng; lưu build SHA, flags, command/log; tích hợp runner chính |
| 3 | Khóa protocol và analysis set | Một phiên bản thống nhất về QP, sequence, scorer, mask, quality metric và multiplicity; amendment có thời điểm rõ |
| 4 | Xác minh A3 | Pin upstream commit/checkpoint hash; ghi đúng conditioning; kiểm tra output; làm rõ mask và fairness |
| 5 | Hoàn thiện E1 đủ phạm vi | Đủ số sequence theo protocol, thêm nội dung độc lập nếu cần; không đếm bản lặp như video mới; báo cáo phạm vi clip/resize |
| 6 | Phân tích lại chất lượng | Absolute và normalized curves, eta, per-sequence/per-QP, overlap, CI, lỗi detector và sensitivity |
| 7 | Đóng gói artifact | CSV nhỏ, config, manifest, environment lock, checksum, lệnh sinh bảng/hình; archive/version cố định, không cần commit video lớn |
| 8 | Viết bản thảo theo bằng chứng | Câu hỏi, related work, phương pháp, kết quả, giới hạn, compute/data/code statement; rà hướng dẫn nơi nộp cụ thể |

Chưa cần triển khai toàn bộ E2/E2t để có một bài E1 tốt. Nếu A1/A3 chưa xong, có thể thu hẹp thành nghiên cứu đánh giá A2, nhưng phải bỏ claim bác bỏ learned pre-editing và claim so sánh với ROI chưa thực hiện. Một negative result chỉ có giá trị thuyết phục khi thiết kế đủ khả năng trả lời câu hỏi; việc không vượt một gate tự đặt không tự bảo đảm có bài quốc tế.

## 9. Biên bản kiểm tra thực thi

Thư viện kiểm tra được cài riêng ngoài repo vào `D:\STUDY\LAB\TEST\.wagur1-review-deps`; không thay requirements hoặc môi trường học thuật của dự án.

| Kiểm tra | Kết quả quan sát |
|---|---|
| Python / NumPy / SciPy / pandas / pytest | 3.12.14 / 2.5.4 / 1.18.1 / 3.1.1 / 9.2.0 |
| Bộ test hiện có | **13 passed in 12.69s**, exit code 0 |
| Biên dịch cú pháp Python, không chạy entrypoint cloud | 20 file trong package/scripts/kaggle/tests, không lỗi cú pháp |
| Fixture e1.csv theo schema runner | `KeyError: 'quality'` |
| Fixture A2 pass, A3 fail | Report trả NO-GO; A2 riêng pass với p=0,00390625 |
| Fixture không có A1 | Report trả NO-GO và gợi ý công bố null + ROI baseline |
| Fixture có A4, ceiling đúng như eval hiện tạo | `KeyError` thiếu ceiling A4 |
| Một quality key bị thiếu, một key bị nhân đôi | `load()` chấp nhận; 120 hàng → 120 hàng |
| BD có quality trùng phát sinh sau gộp rate | PCHIP báo `ValueError` |
| Wilcoxon exact, 5 chênh lệch âm khác nhau | p=0,03125 |
| x264 có/không ROI, mock lệnh | Lệnh encode giống nhau ngoại trừ đầu ra |
| Encode/decode/scorer/SCOPE trên video thật | **Chưa chạy lại trong lần đánh giá này** |

Script tái hiện các probe được lưu ở `D:\STUDY\LAB\TEST\.wagur1-review-probe.py`; các fixture đều là dữ liệu tổng hợp và không phải kết quả nghiên cứu. Chạy từ PowerShell:

```powershell
& 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'D:\STUDY\LAB\TEST\.wagur1-review-probe.py'
```

Test BD đạt không phủ nhận các lỗi tích hợp kể trên: test hiện tại chưa bao phủ schema runner/eval/report, baseline ROI và các trạng thái quyết định. Điểm 4,5/10 cần được chấm lại sau khi sửa pipeline và cung cấp artifact E1; chưa có cơ sở để hứa một mức điểm hoặc khả năng được nhận sau sửa.

## 10. Tài liệu đối chiếu công khai

Các nguồn sau được mở kiểm tra ngày 07/11/2026. Đây là nguồn nền cho nhận xét kỹ thuật; không phải chứng nhận repo đáp ứng yêu cầu một tạp chí cụ thể.

1. [SCOPE: Semantic COdec Pre-Editing for Object-Preserving Compression](https://arxiv.org/abs/2610.39778) và [repo tác giả](https://github.com/miccunifi/SCOPE): xác nhận hướng pre-editing đã có công trình gốc. Không coi adapter là mô hình mới.
2. [A Preprocessing Framework for Video Machine Vision under Compression](https://arxiv.org/abs/2512.15331): tiền lệ về neural preprocessing video và đánh giá bằng codec chuẩn.
3. [The Bjøntegaard Bible](https://arxiv.org/abs/2304.12852): cơ sở đối chiếu về miền so sánh, nội suy và độ nhạy của BD; không dùng bài này để mặc định sai số cho dữ liệu WAGUR1.
4. [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html): giả thuyết kiểm định, xử lý tie/zero và phương pháp tính p.
5. [x265 CLI](https://x265.readthedocs.io/en/master/cli.html) và [x265 API](https://x265.readthedocs.io/en/master/api.html): cấu hình video, vòng đời đối tượng và flush encoder.
6. [Ultralytics COCO converter](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/data/converter.py): mapping giữa chỉ số class và COCO category ID.
7. [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist): tham khảo thực hành về claim, tái lập, thông số thực nghiệm và giới hạn; không áp dụng như luật chung cho mọi tạp chí. Việc thiếu một mục báo cáo cần được giải thích theo ngữ cảnh, không tự quyết định chất lượng khoa học.
