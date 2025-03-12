window.addEventListener("scroll", function () {
  const navbar = document.querySelector(".navbar");
  if (window.scrollY > 50) {
    navbar.classList.add("shrink"); // ถ้าเลื่อนลงมากกว่า 50px ให้ลดขนาด
  } else {
    navbar.classList.remove("shrink"); // ถ้ากลับไปด้านบนให้เป็นเหมือนเดิม
  }
});

// ฟังก์ชันส่งฟอร์มผ่าน AJAX, แล้วแสดงผลใน resultSpan
async function handleFormSubmit(formElem, endpoint, resultSpan, resultKey) {
  const formData = new FormData(formElem);

  // แปลงเป็น URLSearchParams
  let params = new URLSearchParams();
  for (let pair of formData.entries()) {
    params.append(pair[0], pair[1]);
  }

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
    });

    const data = await response.json();
    if (data.error) {
      resultSpan.style.color = "red";
      resultSpan.textContent = "Error: " + data.error;
    } else {
      // ใช้ key ที่ถูกต้องจาก API Response
      let pred = data[resultKey];
      if (pred === undefined) pred = 0;

      resultSpan.style.color = "blue";
      let val = Number(pred).toFixed(2);
      resultSpan.textContent = "Prediction: " + val;
    }
  } catch (err) {
    resultSpan.style.color = "red";
    resultSpan.textContent = "เกิดข้อผิดพลาด";
    console.error(err);
  }
}

// ฟอร์ม Linear Regression
const formLR = document.getElementById("formLR");
const lrResult = document.getElementById("predictResult"); // ✅ แก้ไขให้ตรงกับ HTML
formLR.addEventListener("submit", function (e) {
  e.preventDefault();
  handleFormSubmit(formLR, "/predict_house_lr", lrResult, "lr_prediction");
});

// ฟอร์ม Gradient Boosting
const formGB = document.getElementById("formGB");
const gbResult = document.getElementById("gbResult");
formGB.addEventListener("submit", function (e) {
  e.preventDefault();
  handleFormSubmit(formGB, "/predict_house_gb", gbResult, "gb_prediction");
});
