Bài này sẽ viết lại những gì học được về Course "Sequence Models" week1 trên Coursera

# 1. Why sequence models? Tại sao là mô mình chuỗi
Sequence models là một trong các phần hấp dẫn nhất của Deep Learning. Models như RNN thực hiện nhận diện giọng nói (speech recognition). Hãy xem một số ứng dụng của sequence models:
- Speech recognition: input là audio clip, output là text transcript (cả input và output đều là sequence data - dữ liệu dạng chuỗi, audioclip được chơi theo thời gian, text là chuỗi các từ)
![1](images/SequenceModels_RNN/whymodel/1.png)
- Music generation cũng là một vấn đề với sequence data. Trong trường hợp này chỉ có output `Y` là sequence. Input có thể trống hoặc có thể là số nguyên nào đó chỉ loại nhạc (genre) bạn muốn tạo ra hoặc một số nốt nhác của đoạn nhạc muốn tạo ra.
![2](images/SequenceModels_RNN/whymodel/2.png)
-  Sentiment classification (phân loại quan điểm): input là một sequence (ví dụ một câu nhận xét bộ phim chẳng hạn), output có thể là các số từ 1 đến 5 chỉ sự hài lòng với bộ phim
![3](images/SequenceModels_RNN/whymodel/3.png)
- DNA sequence analysis (phân tích chuỗi DNA): DNA được biểu diễn bằng các chữ cái A, C, G, T, cho một chuỗi DNA chúng ta có thể gán nhãn phần nào của chuỗi DNA tương ứng với mã protein
![4](images/SequenceModels_RNN/whymodel/4.png)
- Machine translation (dịch máy): dịch từ ngôn ngữ này sang ngôn ngữ khác, cả input và output đều ở dạng sequence
![5](images/SequenceModels_RNN/whymodel/5.png)
- Video activity recognition (nhận diện hành động trong video): input là sequence of video frames, yêu cầu cần nhận diện hành động trong các frames đó
![6](images/SequenceModels_RNN/whymodel/6.png)
- Name entity recognition (nhận diện thực thể có tên): Ví dụ input là một câu và cần xác định đâu là tên người trong câu đó. 
![7](images/SequenceModels_RNN/whymodel/7.png)

Tất cả các bài toán trên đầu có thể giải quyết như supervised learning (học giám sát). Nhận thấy example X và label Y có thể là các loại sequence khác nhau, độ dài có thể khác nhau, có trường hợp chỉ X hoặc Y là sequence