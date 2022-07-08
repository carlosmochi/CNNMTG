
var searchFunctions = {
    scryRequest: new XMLHttpRequest(),
    SCRYURL: "https://api.scryfall.com/cards/",
    testing:function(){
        console.log("working")
    },
    getCardbyId: function(cardId){
        searchFunctions.searchIMGID(cardId).then(response =>{
            setTimeout(()=>{

                var img = document.getElementById("cardIMG")
                img.removeAttribute("src")
                console.log(response)
                console.log(response.card_faces)

                if(response.card_faces === undefined){
                    img.setAttribute("src", response.image_uris.normal)
                    document.getElementById("cardName").innerText = response.name
                    document.getElementById("cardCMC").innerText = response.mana_cost
                    document.getElementById("cardType").innerText = response.type_line
                    document.getElementById("cardFlavor").innerText = response.flavor_text
                    document.getElementById("cardText").innerText = response.oracle_text
                }else{
                    img.setAttribute("src", response.card_faces[0].image_uris.normal)
                    document.getElementById("cardName").innerText = response.card_faces[0].name
                    document.getElementById("cardCMC").innerText = response.card_faces[0].mana_cost
                    document.getElementById("cardType").innerText = response.card_faces[0].type_line
                    document.getElementById("cardFlavor").innerText = response.card_faces[0].flavor_text
                    document.getElementById("cardText").innerText = response.card_faces[0].oracle_text
                }
                if(response.flavor_text === undefined){
                    document.getElementById("cardFlavor").innerText = "Flavorless"
                }

            }, 1000)
        })
    },
    searchIMGID: function(cardId){
        return new Promise((resolve) => {
            console.log("working")
            var searchQuerry = this.SCRYURL + cardId;
            this.scryRequest.open("GET", searchQuerry, true)
            this.scryRequest.responseType = 'json'
            this.scryRequest.onload = () => {
                resolve(this.scryRequest.response)
            }
            this.scryRequest.send()
        });
    },
    // getImagebyQuery: function(cardQ){
    //     this.searchIMGQuery(cardQ).then(response =>{
    //         console.log(response.image_uris.border_crop)
    //     })
    // },
    searchIMGQuery: function (cardQuery) {
        return new Promise((resolve) => {
            console.log("working")
            var searchQuery = this.SCRYURL + "/search?q=" + cardQuery;
            this.scryRequest.open("GET", searchQuery, true)
            this.scryRequest.responseType = 'json'
            this.scryRequest.onload = () => {
                resolve(this.scryRequest.response)
            }
            this.scryRequest.send()
        });
    },
    getCardbyQuery: function(){
        var cardQ = document.getElementById("cardQuery").value
        this.searchIMGQuery(cardQ).then(response =>{
            console.log(response)
            var img = document.createElement("img")
            var imgSpace = document.getElementById("cardImage")
            this.resetHTML()
            if(response.data.length > 1){
                var ul = document.createElement("ul")
                ul.setAttribute("id","imageList")
                imgSpace.appendChild(ul)
                for(var i = 0; i < response.data.length; i++){
                   // var li = document.createElement("li")
                    img = document.createElement("img")
                    img.setAttribute("id", "cardIMG"+i)
                    img.setAttribute("src",response.data[i].image_uris.normal)
                    //li.appendChild(img)
                    document.getElementById("imageList").appendChild(img)
                }
            }else {
                img.setAttribute("id", "cardIMG")
                img.setAttribute("src", response.data[0].image_uris.normal)
                imgSpace.appendChild(img)
                document.getElementById("cardName").innerText = response.data[0].name
                document.getElementById("cardCMC").innerText = response.data[0].mana_cost
                document.getElementById("cardType").innerText = response.data[0].type_line
                document.getElementById("cardFlavor").innerText = response.data[0].flavor_text
                document.getElementById("cardText").innerText = response.data[0].oracle_text
            }
        })
    },
    resetHTML:function(){
        var imgSpace = document.getElementById("cardImage")
        while(imgSpace.firstChild != null){
            imgSpace.removeChild(imgSpace.firstChild)
        }
    }

}
window.searchFunctions = searchFunctions;
export {searchFunctions};
