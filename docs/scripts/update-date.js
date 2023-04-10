const updateTextCell = document.querySelector(".update-date");

var xhttp = new XMLHttpRequest();
xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
        let data = JSON.parse(this.responseText);
        let update_date = new Date(data.updated_at);
        // updateTextCell.textContent = 'last updated: ' + update_date.toISOString();
        let date_text = pad(update_date.getFullYear(), 4);
        date_text = date_text + '-' + pad(update_date.getMonth(), 2);
        date_text = date_text + '-' + pad(update_date.getDate(), 2);
        date_text = date_text + ' ' + pad(update_date.getHours(), 2);
        date_text = date_text + ':' + pad(update_date.getMinutes(), 2);
        date_text = date_text + ':' + pad(update_date.getSeconds(), 2);
        updateTextCell.textContent = 'last update: ' + date_text;
    }
};
xhttp.open("GET", "https://api.github.com/repos/v-pnk/cadloc", true);
xhttp.send();

function pad(num, size) {
    num = num.toString();
    while (num.length < size) num = "0" + num;
    return num;
}