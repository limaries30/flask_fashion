//========================================================================
// Drag and drop image handling
//========================================================================

var fileDrag = document.getElementById("file-drag");
var fileSelect = document.getElementById("file-upload");

// Add event listeners
fileDrag.addEventListener("dragover", fileDragHover, false);
fileDrag.addEventListener("dragleave", fileDragHover, false);
fileDrag.addEventListener("drop", fileSelectHandler, false);
fileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
  // prevent default behaviour
  e.preventDefault();
  e.stopPropagation();

  fileDrag.className = e.type === "dragover" ? "upload-box dragover" : "upload-box";
}

function fileSelectHandler(e) {
  // handle file selecting
  var files = e.target.files || e.dataTransfer.files;
  fileDragHover(e);
  for (var i = 0, f; (f = files[i]); i++) {
    previewFile(f);
  }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var imagePreview = document.getElementById("image-preview");
var imageDisplay = document.getElementById("image-display");
var uploadCaption = document.getElementById("upload-caption");
var predResult = document.getElementById("pred-result");

var predUpper = document.getElementById("pred-upper");
var predLower = document.getElementById("pred-lower");
var predDress=document.getElementById("pred-dress");

var loader = document.getElementById("loader");

var UPPER_CLOTHES=['BG',
 'short_sleeved_shirt',
 'long_sleeved_shirt',
 'short_sleeved_outwear',
 'long_sleeved_outwear',
 'vest',
 'sling',]
var LOWER_CLOTHES=[
 'shorts',
 'trousers',
 'skirt']
var DRESS=[
 'short_sleeved_dress',
 'long_sleeved_dress',
 'vest_dress',
 'sling_dress']
//========================================================================
// Main button events
//========================================================================

function submitImage() {
  // action for the submit button
  console.log("submit");

  if (!imageDisplay.src || !imageDisplay.src.startsWith("data")) {
    window.alert("Please select an image before submit.");
    return;
  }

  loader.classList.remove("hidden");
  imageDisplay.classList.add("loading");

  // call the predict function of the backend
  predictImage(imageDisplay.src);
}

function clearImage() {
  // reset selected files
  fileSelect.value = "";

  // remove image sources and hide them
  imagePreview.src = "";
  imageDisplay.src = "";

  predResult.innerHTML = "";
  predUpper.innerHTML = "";
  predLower.innerHTML = "";
  predDress.innerHTML = "";

  hide(imagePreview);
  hide(imageDisplay);
  hide(loader);
  hide(predResult);
  hide(predUpper);
  hide(predLower);
  hide(predDress);
  show(uploadCaption);

  imageDisplay.classList.remove("loading");
}



function previewFile(file) {
  // show the preview of the image
  console.log(file.name);
  var fileName = encodeURI(file.name);

  var reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    imagePreview.src = URL.createObjectURL(file);

    show(imagePreview);
    hide(uploadCaption);

    // reset
    predResult.innerHTML = "";
    predUpper.innerHTML = "";
    predLower.innerHTML = "";

    imageDisplay.classList.remove("loading");

    displayImage(reader.result, "image-display");
  };
}

//========================================================================
// Helper functions
//========================================================================

function predictImage(image) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(image)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {

          displayImage(data.image_response,"image-display");
          displayResult(data);
    
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayImage(image, id) {
  // display image on given id <img> element
  let display = document.getElementById(id);
  display.src = image;
  show(display);
}

function displayResult(data) {
  // display the result
  // imageDisplay.classList.remove("loading");
  hide(loader);
  imageDisplay.classList.remove("loading");
  var obj=data.result;

  Object.keys(obj).forEach(function(key) {

    if(UPPER_CLOTHES.includes(key))
    {
    predUpper.innerHTML += key + " upper " + obj[key];
    show(predUpper);
    }
    
    if(LOWER_CLOTHES.includes(key))
    {
    predLower.innerHTML += key + " lower " + obj[key];
    show(predLower);
    }

    
    if(DRESS.includes(key))
    {
    predDress.innerHTML += key + " lower " + obj[key];
    show(predDress);
    }

    });

 
}

function hide(el) {
  // hide an element
  el.classList.add("hidden");
}

function show(el) {
  // show an element
  el.classList.remove("hidden");
}