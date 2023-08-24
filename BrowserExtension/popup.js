  // Add an event listener to receive messages from the background script
browser.runtime.onMessage.addListener((message) => {
  console.log(message);
  // Perform actions based on the received message
});