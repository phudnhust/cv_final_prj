import face_alignment
from skimage import io
import torch
# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

# input = io.imread('/mnt/HDD1/phudh/course/img_manipulation/final/DiffSwap/test.png')
# preds = fa.get_landmarks(input)
# print(preds)

import torch

# Tạo một tensor với requires_grad=True
x = torch.tensor([2.0], requires_grad=True)

# Thực hiện một phép toán trong torch.no_grad()
with torch.no_grad():
    y = x * 3  # Phép toán không có gradient

# Thực hiện một phép toán có gradient
z = y ** 2

# Tính đạo hàm của z đối với x
# Lỗi xuất hiện ở đây vì x không còn có gradient (do torch.no_grad() ở trên)
grad_x = torch.autograd.grad(z, x)[0]
