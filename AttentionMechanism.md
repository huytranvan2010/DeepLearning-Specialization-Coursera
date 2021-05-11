Sequen models có thể được tăng cường bằng `attention mechasnism`. Algorithm này giúp cho model biết chỗ nào nó nên tập trung vào của sequence of inputs.

# Various sequence to sequence architectures
## Basic models
![1](images/AttentionMechanism/1.png)
![2](images/AttentionMechanism/2.png)

## Picking the most likely sentence
### Finding the most likely transaletion - tìm câu dịch thích hợp nhất

Machine translation này giống conditional language model. Language model đi tạo ra các câu, tính xác suất của câu được tạo ta. Machine translation cũng tạo ra câu dịch tuy nhiên đi tính xác suất của câu dịch với điều kiện có câu input cần dịch. 

Đầu vào của language model là vector a<0>=0 còn đầu vào của machine translation có thể coi là encoding vector biểu diễn cho input sentence.
![3](images/AttentionMechanism/3.png)

Việc chọn ngẫu nhiên câu dịch có thể đúng có thể sai (nghĩa hoàn toàn khác nhau). Do đó cần có algorithm để xác định max của xác suất có điều kiện.
![4](images/AttentionMechanism/4.png)

### Vì sao không phải là Greedy Search

Greedy search không phải lúc nào cũng cho kết quả tốt nhất, nhìn ví dụ trong hình sẽ thấy. Nếu `vocab_size=10000`, một câu dịch có 10 từ, khi đó sẽ có 10000^10 số câu có thể tạo ra. Việc tìm kiếm và thử số câu này là không tưởng, do đó cần tìm kiếm cách thích hợp để chọn được câu có thể chấp nhận được mà không quá chậm
![5](images/AttentionMechanism/5.png)
