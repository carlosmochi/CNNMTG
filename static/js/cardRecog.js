import {searchFunctions} from "./ScryfallIntegration.js";

var cardRecog={
    sendImage: function(dataToServer){
        var request = new XMLHttpRequest()
        var sendData = {
            data: dataToServer
        }
        request.open('POST', '/readImage', false)
        request.setRequestHeader("Content-Type", "application/json;charset=UTF-8")
        request.send(JSON.stringify(sendData))

    },
    getResponse: function(dataToServer){
        this.sendImage(dataToServer)
        var request = new XMLHttpRequest();
        return new Promise(resolve => {
            request.open('GET','/readImage', true)
            request.onload = function (e) {
                console.log("chegou no GET")
                resolve(request.response)
            }
            request.send()
        })
    },
    testing:function (data) {
        this.getResponse(data).then(response =>{
            console.log("got to final")
            console.log(response)
            this.hideCamera()
            searchFunctions.getCardbyId(response)
        })
    },
    hideCamera:function (){
        console.log("working")
        document.getElementById("videoElement").setAttribute("hidden", "true")
    },
    showCamera:function (){
        document.getElementById("videoElement").removeAttribute("hidden")
        document.getElementById("cardIMG").removeAttribute("src")
    },
    runCamera:function() {
        var video = document.querySelector("#videoElement");
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({video: true})
                .then(function (stream) {
                    video.srcObject = stream;
                })
                .catch(function (err0r) {
                    console.log("Something went wrong!");
                    console.log(err0r)
                });
        }
    },
    capturedImage:function () {
        var canvas = document.getElementById('canvas');
        var video = document.getElementById('videoElement');
        canvas.removeAttribute("hidden")
        canvas.width = video.getBoundingClientRect().width;
        canvas.height = video.getBoundingClientRect().height;
        canvas.getContext('2d').drawImage(video, 0, 0);
        this.hideCamera()
    },
    captureB64Image: function(){
        var canvas = document.getElementById('canvas');
        var video = document.getElementById('videoElement');
        canvas.removeAttribute("hidden")
        canvas.width = video.getBoundingClientRect().width;
        canvas.height = video.getBoundingClientRect().height;
        canvas.getContext('2d').drawImage(video, 0, 0);
        var b64Image = canvas.toDataURL();
        canvas.height = 0
        canvas.width = 0
        b64Image = this.convertDataURLtoB64(b64Image)
        return b64Image
    },
    convertDataURLtoB64:function (image) {
        var b64 = image
        b64.replace('data:', '').replace(/^.+,/, '');
        return b64
    }
}
window.cardRecog = cardRecog;
export {cardRecog};