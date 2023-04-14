
function changeImage(input) {
var reader;

if (input.files && input.files[0]) {
  imgChanged=true
 reader = new FileReader();

 preview.hidden=false;

 reader.onload = function(e) {
   preview.setAttribute('src', e.target.result);
 }

 reader.readAsDataURL(input.files[0]);
 var generate= document.getElementById('enter')
 generate.hidden=false

 if(imgChanged){
  const promise1 =new Promise((resolve, reject)=>{
    resolve(storeLocalStorage(preview, flag=1));
  });
  
  const promise2= promise1.then((resolve, reject)=>{
  setTimeout(() =>triggerApiCall(), 2000)
               });
}
}
}

function openModal(img,fp){
          //custom modal
          var btn = document.getElementById("cx");
          var modal = document.getElementById("myModal");
          btn.onclick = function() {
            modal.style.display = "block";
          }

          // Get the <span> element that closes the modal
          var span1 = document.getElementsByClassName("close")[0];

          // When the user clicks anywhere outside of the modal, close it
          window.onclick = function(event) {
            if (event.target == modal) {
              modal.style.display = "none";
            }
          }

          var top=document.getElementById('top')
          top.src = img.src;
          top.hidden=false;
          storeLocalStorage(top, flag=0)
          var temp=document.getElementById('temp')
          temp.hidden=false

          var down=document.getElementById('download')
          down.hidden=false

          var preview = document.getElementById("preview");
          var previewOutput = document.getElementById("previewOutput");
          var arrow=document.getElementById('enter')
          
          document.getElementById('upload').hidden = false;

          var fileTag = document.getElementById("upload");
          
         fileTag.addEventListener("change", function() {
         changeImage(this);
         });

         preview.hidden=false;

}

function storeLocalStorage(elem, flag){
  elem.addEventListener("load", function () {
    var imgCanvas = document.createElement("canvas"),
        imgContext = imgCanvas.getContext("2d");

    // Make sure canvas is as big as the picture
    imgCanvas.width = elem.width;
    imgCanvas.height = elem.height;

    // Draw image into canvas element
    imgContext.drawImage(elem, 0, 0, elem.width, elem.height);

    // Get canvas contents as a data URL
    var imgAsDataURL = imgCanvas.toDataURL("image/png");
    try{    flag===0?
      localStorage.setItem("cloth_img", imgAsDataURL):
      localStorage.setItem("uploaded_person_img", imgAsDataURL) ;}
     catch (e) {
      console.log("Storage failed: " + e);
      }
    }, false
    )
}

function do_close(){
  var modal = document.getElementById('myModal');
  window.location.reload();
  if(localStorage) { // Check if the localStorage object exists
    // console.log("clearing...")
    // document.getElementById('preview').src = "";
    // document.getElementById('top').src = "";
    document.getElementById('previewOutput').src = "";
    // var closeBtn = document.getElementById('close');
    // closeBtn.addEventListener('click', function() {
    //   Cache.delete()
    modal.hidden = true;
    //   op.remove();
    // });
    document.getElementById('arrow').remove();
    preview.remove();
    document.getElementById('previewOutput').remove();
    modal.remove();
    localStorage.clear()  //clears the localstorage

} else {

    alert("Sorry, no local storage."); //an alert if localstorage is non-existing
}
}

function myFunction() {
  var element = document.body;
  element.classList.toggle("dark-mode");
  var modal = document.getElementById("modal-content");
  modal.style.backgroundColor="#0F3D3E"

}

 function triggerApiCall(){
  const cloth_img = localStorage.getItem('cloth_img');
  const uploaded_person_img = localStorage.getItem('uploaded_person_img');
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload-image');
  xhr.setRequestHeader('Content-Type', 'application/json');
  const data = { 'cloth_img': cloth_img , 'uploaded_person_img':uploaded_person_img};
  const jsonData = JSON.stringify(data);
 xhr.send(jsonData);
}

function getDataFromFlaskAPI(){
  const url = 'http://localhost:5000/get-processed-image'
  // const response = fetch(url)
  // console.log(response);

  $.ajax(
    {
        type: 'GET',
        url: url,
        success: function (data) {
            if (data === undefined || data.length == 0) {
                console.log("failed");
            }
            console.log("data is", data.response);
            document.getElementById("previewOutput").setAttribute('src' ,data.response);

        },
        error: function (textStatus, errorThrown) {
            alert("Error!!")
        }
    }
);
  }

  function downloadData(){
    //const divElement = document.getElementById('preview');
    const imgElement = document.getElementById('previewOutput'); // Get the first img element inside the div
    const imageUrl = imgElement.src; // Get the URL of the image
    const linkElement = document.createElement('a'); // Create a new 'a' element
    console.log(linkElement)
    linkElement.setAttribute('href', imageUrl);
    linkElement.setAttribute('download', 'myImage.jpeg');
    linkElement.click();
  }
