// function openSideNav() {
//   document.getElementById("sideNavigation").style.width = "200px";
// }
//
// /* Set the width of the side navigation to 0 */
// function closeSideNav() {
//   document.getElementById("sideNavigation").style.width = "0";
// }

var sideNavigation ={
  openSideNav: function (){
    document.getElementById("sideNavigation").style.width = "200px";
    document.getElementById("screenSpace").style.marginLeft = "195px";
  },
  closeSideNav: function (){
    document.getElementById("sideNavigation").style.width = "0";
    document.getElementById("screenSpace").style.marginLeft = "0";
  }
}
window.sideNavigation = sideNavigation;
export {sideNavigation};
