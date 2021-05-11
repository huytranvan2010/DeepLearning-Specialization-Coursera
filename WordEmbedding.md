# Word Embedding


## Word representation
Nếu chỉ biểu diễn các word dạng one-hot vector thì `distance` giữa các từ là như nhau, không có sự khác biệt. Điều này sẽ không tốt nếu trong trường hợp chúng ta dự đoán từ.
![1]("images/WordEmbedding/1.png")
Ví dụ như hình trên nhận thấy `orange` và `apple` tương đồng thì có thể dễ dàng dự đoán từ phía sau `apple` là `juice`.

Các từ được biểu diễn bằng vector, do đó độ tương đồng giữa chúng có thể được biểu diễn qua `cosine similarity`

## Embedding matrix

Muốn thực hiện được `word embedding` thì chúng ta cần học được `embedding matrix`.

Ví dụ vocabulary có `vocab_size` là 10000. Khi đó mỗi từ được biểu diễn bằng one-hot vetor (10000, 1). Để có thể biểu diễn từ theo `word embedding` với chiều 300, chúng ta sẽ cần `embedding matrix` có chiều (300, 10000). Khi đó lấy tích `embedding matrix` với one-hot vector ta được embedding của từng từ. 

Nhận thấy embedding của từ trùng với cột của `embedding matrix` có cùng chỉ số.

