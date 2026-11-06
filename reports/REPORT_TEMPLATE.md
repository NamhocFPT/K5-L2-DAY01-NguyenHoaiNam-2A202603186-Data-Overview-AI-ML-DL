# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 11/6**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh**. Task là `image_classification`, nên model
  gán nhãn cho toàn ảnh. Record không có bounding box hay tọa độ nào, chỉ có
  kích thước ảnh (640x428), và rank 1–5 là các lớp ứng viên cho cùng một ảnh
  chứ không phải các vật thể khác nhau. Vì vậy `cab` (0.51) là dự đoán cho
  cả ảnh `traffic`, không phải chỉ một chiếc taxi cụ thể trong ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp **đi kèm taxonomy/checkpoint**, không do model tự nghĩ ra. Model `yolo11n-cls.pt` được huấn luyện với bộ nhãn ImageNet-1K (1000 lớp cố định). Model chỉ chọn một lớp trong danh sách này, nên `class_id` 468 chỉ có nghĩa là "cab" trong bảng nhãn ImageNet-1K.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Chỉ ghi tên lớp là chưa đủ. `class_id` là chỉ số để máy xử lý, `class_name` giúp người đọc hiểu, còn `taxonomy_name` cho biết id đó thuộc bảng nhãn nào (cùng một id ở taxonomy khác có thể là lớp khác). Ghi đủ ba trường thì kết quả không mơ hồ, kiểm tra lại được và tái lập được.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần nêu rõ quy tắc chọn **một** lớp khi ảnh có nhiều vật: chọn chủ thể chính (vật nổi bật nhất hoặc chiếm diện tích lớn nhất), cách xử lý khi nhiều chủ thể ngang nhau, và cách xử lý khi không lớp nào phù hợp. Nếu không muốn ép một lớp thì phải chuyển sang multi-label. Nhờ vậy những người gán nhãn khác nhau sẽ cho kết quả nhất quán.  
6. Vì sao không được lấy `score` của model làm ground truth?
- `score` chỉ cho biết model tự tin đến mức nào, không cho biết dự đoán có đúng hay không. Model có thể tự tin mà vẫn sai, ví dụ ảnh `kitchen` bị dự đoán là `gong` (0.42) và ảnh COCO lệch domain so với ImageNet. Ground truth phải do người gán nhãn theo guideline, nếu lấy `score` của model làm ground truth thì model đang tự chấm bài của mình
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
 Cái box là một khung chữ nhật bao quanh người đứng quay lưng ở bên phải ảnh.
   - Góc trên-trái: cách mép trái ảnh 385.33 pixel và cách mép trên ảnh 69.24
     pixel (gần chỗ đầu người).
   - Góc dưới-phải: cách mép trái ảnh 498.92 pixel và cách mép trên ảnh 348.92
     pixel (gần chỗ bàn chân).
   - Kích thước: rộng 113.58 pixel (498.92 − 385.33) và cao 279.68 pixel
     (348.92 − 69.24), nên khung hẹp và dài theo chiều đứng, vừa với dáng người.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
 Ở ngưỡng 0.35 ảnh kitchen có 11 prediction, ở ngưỡng 0.60 chỉ còn 6.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Hạ ngưỡng thì giữ lại nhiều hộp hơn, trong đó có nhiều hộp điểm thấp dễ sai. Reviewer phải kiểm tra thêm từng hộp: xóa hộp thừa hoặc nhầm vật (falsepositive), sửa nhãn sai, chỉnh hộp chưa khít, và bổ sung hộp còn thiếu. Khối lượng review tăng lên, đổi lại ít bỏ sót vật hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Quy tắc: vẽ **một hộp chữ nhật khít cho mỗi vật thể**, bao trọn phần nhìn thấy được của vật từ mép ngoài cùng bên trái đến phải và từ trên xuống dưới, không chừa nhiều nền thừa và không cắt mất một phần vật. Hai vật khác nhau thì vẽ hai hộp riêng.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Ví dụ: hộp `person` (0.61) ở mép trái ảnh kitchen chỉ thấy một phần cánh tay và bàn tay, phần còn lại bị khung ảnh cắt mất.
   - Phần **viết thành quy tắc trong guideline**: với vật bị cắt mép hoặc bị che,
     chỉ vẽ hộp quanh phần nhìn thấy được và để hộp chạm mép ảnh. Quy tắc này
     lặp lại ở rất nhiều ảnh, nên cần ghi sẵn để mọi người gán nhãn làm giống nhau.
   - Phần **đưa lên hỏi Lab Coach (escalation)**: chỉ thấy một bàn tay thì có
     còn tính là `person` không, và phải thấy tối thiểu bao nhiêu phần thì mới
     gán nhãn. Đây là quyết định chính sách, không tự suy ra được từ ví dụ, nên
     cần Lab Coach quyết định rồi mới ghi vào guideline.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ là một hình chữ nhật bao quanh người, nên bên trong còn lẫn nhiều nền(tường, nồi chảo, tủ). Polygon đi theo đường viền thật của vật: có đầu, vai, hai tay chống hông và hai chân tách ra.
Nhờ vậy biết chính xác pixel nào thuộc người, pixel nào là nền, và tính được hình dạng, diện tích thật của vật. Với kitchen-001, polygon có 348 điểm nên ôm sát dáng người, còn box (khoảng 113 x 282 px) thì không phân biệt được.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` (ví dụ kitchen-001) là tên riêng để phân biệt từng đối tượng (từng mask) trong cùng một ảnh, giúp nói chính xác "mask nào" khi review, sửa hoặc báo lỗi. Hai bowl cùng lớp vẫn có hai `instance_id` khác nhau.
   - Nó **không phải class ID**: class ID (như 0 = person) cho biết vật thuộc
     loại nào, nhiều vật cùng loại dùng chung một class ID, còn `instance_id`
     là duy nhất cho từng vật.
   - Nó **không phải tracking ID**: tracking ID theo dõi cùng một vật qua nhiều
     khung hình video, còn `instance_id` chỉ đánh số các vật trong một ảnh
     tĩnh và không có ý nghĩa giữa các ảnh khác nhau.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Quy tắc: vẽ biên mask bám sát mép ngoài của phần nhìn thấy được của vật, sao cho mask phủ toàn bộ vật nhưng không lấn sang nền hay vật khác. Mỗi vật một mask riêng. Chỗ bị che bởi vật khác thì không đưa phần che vào mask.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
 Ví dụ: trên ảnh kitchen, `dining table` chạm và bị các bowl, chai mứt, tấm bột đặt đè lên, còn biên với `oven` (phía trên-trái) chồng lấn nhau, nên khó biết mép bàn kết thúc ở đâu.
   - Dùng **guideline** cho trường hợp lặp lại và có thể mô tả bằng quy tắc: "vật nằm trên bàn thì mask bàn không bao gồm vật đó, mỗi vật một mask riêng, vùng hai vật chồng nhau thuộc về vật ở phía trước". Quy tắc này áp dụng cho rất nhiều ảnh nên cần ghi sẵn để mọi người làm giống nhau.
   - **Escalation** (hỏi Lab Coach) chỉ khi guideline không trả lời được:ví dụ mép bàn bị bột phủ nên không rõ đâu là bàn, đâu là bột, hoặc vùng tối giữa oven và bàn không xác định được là vật nào. Đây là tình huống mơ hồ cần người có thẩm quyền quyết định rồi cập nhật guideline.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |1 nhãn lớp cho cả ảnh|kitchen top-1 là gong (0.42), sai; traffic là cab (0.51) dù có nhiều xe, ImageNet không có lớp “traffic”|Chọn 1 lớp đúng theo guideline (chủ thể chính)|Nhãn có khớp nội dung chính và đúng taxonomy không|
| Phát hiện vật thể |1 box + 1 lớp cho mỗi vật|kitchen: person 0.61 và oven 0.69 bị cắt mép trái; nhiều bowl nhỏ điểm thấp (0.38 đến 0.46)|Vẽ box khít phần nhìn thấy, thêm box thiếu, xóa box thừa|Box có khít không, thiếu/thừa box không, lớp đúng không|
| Instance segmentation |1 mask (polygon) + lớp + instance_id cho mỗi vật|kitchen-005 (dining table) bị bowl, chai, bột đè lên, biên không rõ và chồng với oven; polygon có điểm lạc|Vẽ biên bám mép vật, mỗi vật một mask, vùng chồng gán cho vật phía trước; mơ hồ thì escalate|Mask có ôm sát không, có lấn nền/vật khác không, tách đúng từng vật không|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ dùng đúng các ảnh mẫu (traffic, kitchen, dining) và file kết quả của lab này; không tải ảnh hoặc dữ liệu ra ngoài phạm vi, không chia sẻ chúng với người ngoài hay công cụ khác, và không đưa thông tin cá nhân nào vào báo cáo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Mình sẽ dừng ngay, không xử lý hay lưu thêm dữ liệu đó, rồi báo cho Lab Coach để họ quyết định cách xử lý. Nếu lỡ push phải báo cáo ngay để có hành vi khác phục
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
