var catalogControl = {
    catalogList: [],
    cardNames: [],
    cardAmount: [],
    cardsInList: 0,
    addToCatalog: function (){
        var alreadyInList = false
        if(document.getElementById("cardName").innerText.length > 0){
            var cardName = document.getElementById("cardName").innerText
            var cardPos = 0
            for(var card = 0;card < this.cardNames.length;card++){
                if(cardName == this.cardNames[card]) {
                    alreadyInList = true
                    cardPos = card
                }
            }
            if(!alreadyInList) {
                this.cardAmount.push(1)
                this.cardNames.push(cardName)
            }else{
                this.cardAmount[cardPos]++
            }
            this.cardsInList++
            window.alert(cardName+" inserido com sucesso. Quantidade de cartas no catalogo: "+this.cardsInList)
        }else{
           window.alert("Não existe carta para ser inserida")
        }
    },
    downloadCatalog: function () {
        for(var card = 0; card < this.cardNames.length;card++){
            this.catalogList.push(this.cardNames[card]+" X"+this.cardAmount[card]+"\n")
        }
        var element = document.createElement('a');
        element.setAttribute('href', 'data:text/plain;,' + encodeURIComponent(this.catalogList.toString()));
        element.setAttribute('download', "Catalogo_CNN.txt");

        element.style.display = 'none';
        document.body.appendChild(element);

        element.click();

        document.body.removeChild(element);

        this.catalogList = [];
    }

}

window.catalogControl = catalogControl;
export {catalogControl};