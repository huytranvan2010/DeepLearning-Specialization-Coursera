# MobileNet
Tính toán nhẹ hơn, có thể triển khai trên các thiết bị di động.

## Motivation for MobileNets
* Chi phí tính toán thấp khi deployment
* Hữu ích cho các ứng dụng di động và hệ thống nhúng
* Key idea: Normal and depthwise-separable convolutions

## Normal convolution
![mo](images/MobileNet/mo0.png)
Giả sử có ảnh `6x6x3`, áp convolution layer vào filter có kích thước `3x3x3` với 5 filters. Khi đó computional cost sẽ được tính như trên  (gần đúng). Mỗi filter có số parameters là `3x3x3` (mỗi lần tính sẽ tính 27 phép tính rồi cộng lại). Filter này di chuyển tất cả `4x4` lần để tạo output. Nên nhớ có 5 filters nên cần nhân thêm với 5. Đây mới chỉ là tính toán cho một ảnh cực kỳ nhỏ. 

## Depthwise Seperable Convolution
Có 2 steps: depthwise convolution + pointwise convolution

### Depthwise convolution
![mo](images/MobileNet/mo1.png)
Ảnh ban đầu `n x n x n_C`, kích thước của filter là `f x f x n_C`. Tuy nhiên khác với `normal convolution` ở `depthwise convolution` sẽ thực hiện convolution operation trên mỗi channels `n x n` của ảnh, và filter là `f x f`. Khi đó sẽ thực hiện `n_C` lần. Đầu ra sẽ được output với số channels là `n_C`.

Cùng xem chi tiết từng channel một.

Đối với the first filter.
![mo](images/MobileNet/mo2.png)

Đối với the second filter.
![mo](images/MobileNet/mo3.png)

Đối với the third filter.
![mo](images/MobileNet/mo4.png)
Nhận thấy ouput có kích thước là `n_out x n_out x n_C`.

![mo](images/MobileNet/mo5.png)
Trên đây là computational cost cho `depthwise convolution`, lúc này chỉ còn `(3 x 3) x (4 x 4) x 3`. Ở đây 3x3 là số phép tính cho một ví trí của output (1 channel), có 4 x 4 vị trí cho 1 channels và tổng cộng có 3 channels.

### Pointwise convolution
![mo](images/MobileNet/mo6.png)

![mo](images/MobileNet/mo7.png)
Pointwise vonvolution - áp dụng `1x1` convolution, khi đó có thể thay đổi số channels of output tùy ý. Kích thước heigh x width đã được xác định ở bước `depthwise convolution` rồi nên không cần lo lắng nữa. Nhận thấy dễ dạng tính được computational cost cho pointwise convolution.

## Cost summary
So sánh computational cost của normal convolution và depthwise seperable convolution.
![mo](images/MobileNet/mo8.png)
Bên trái là công thức tổng quát so sánh computational cost của depthwise seperable convolution với normal convolution. Nói chung `depthwise seperable convolution` nhanh hơn khoảng 10 lần so với `normal convolution`.

![mo](images/MobileNet/mo9.png)
Chú ý về cách biểu diễn. Filter ở phần depthwise và pointwise convolution không biểu diễn chính xác số channels (chỗ này chỉ biểu diễ tượng trưng thôi).




