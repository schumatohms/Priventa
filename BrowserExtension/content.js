 
//code from https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Your_second_WebExtension
browser.runtime.onMessage.addListener((message) => {
  console.log(message);
  // Display the message to the user
  document.getElementById("message").textContent = message.message;
});