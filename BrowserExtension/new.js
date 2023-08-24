//Declare constand for the web api
const flaskAPI = 'http://127.0.0.1:5000/predict'

 //check when a browser tab is created
  //codes below from https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/query
  function logTabs(tabs) {

    console.log('this na browser tab own' + tabs[0].url);
  }
  
  function onError(error) {
    console.error(`Error: ${error}`);
  }
  
  browser.tabs
    .query({ currentWindow: true, active: true })
    .then(logTabs, onError);
  
  
// Add event listener for tab activation
// code from https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/webNavigation/onBeforeNavigate#details_2
browser.webNavigation.onBeforeNavigate.addListener((details) => {
  fetch(details.url)
  .then(response => {           
        return response.text();
  })
  .then(text => {                       
                // Handle HTML content
        const htmlContent = text;  

        // Create a DOMParser to parse the HTML
     //code ideas from https://developer.mozilla.org/en-US/docs/Web/API/DOMParser
        const parser = new DOMParser();
        const htmlDoc = parser.parseFromString(htmlContent, 'text/html');

        // Find all <script> elements within the HTML
        const scriptElements = htmlDoc.querySelectorAll('script');

        // Extract inline script contents
        const inlineScripts = [];
        for (const scriptElement of scriptElements) {
            //Skip external scripts
            if (!scriptElement.src) {
                // This is an inline script
                inlineScripts.push(scriptElement.textContent);
            }
        }
        if (inlineScripts.length > 0) { 
            //console.log('inline script is' + inlineScripts);
            //call predict function
          
          predict(inlineScripts)
            .then(result => {
                // Use the result in this function
                console.log('Value of inline resp:', result);
                if (result == 'yes') {
                  console.log('siteblocked');
                 browser.tabs.update(details.tabId, { url: 'background.html' });
                 browser.tabs.sendMessage(details.tabId, { message: 'Site blocked', url: details.url });
                  //return { cancel: true };                  
                }
            })
            .catch(error => {
                // Handle errors from the predict function
                console.error('Error from predict:', error);
            });
            //console.log('the response from inline is ' + c);             
        }

})
  .catch(err => console.error(err));
});
//Web request for external javascript files
browser.webRequest.onBeforeRequest.addListener(
  function (details) { 
      if (details.url.endsWith('.js')) { 

           fetch(details.url)
              .then(response => {        
                   return response.text(); 
              })
              .then(text => {
             
              predict(text)
                .then(result => {
                  // Use the result in this function
                  console.log('Value of external resp:', result);
                  if (result == 'yes') {
                    console.log('siteblocked');
                  browser.tabs.update(details.tabId, { url: 'background.html' });
                  browser.runtime.sendMessage({ message: 'Site blocked', url: details.url });
                  return { cancel: true };                  
                  }
                })  
                .catch(error => {
                    // Handle errors from the predict function
                    console.error('Error from predict:', error);
                });                              
              })
              .catch(err => console.error(err));
      }
     // return new Promise(resolve => setTimeout(resolve, 2000));
  },
{urls: ['<all_urls>']},
  ['blocking']
);

 browser.webNavigation.onCompleted.addListener(function(details) {
  console.log('page loading completed')
});

//predicT function sending and receiving responses from the web API
function predict(scripts) {
    return fetch(flaskAPI, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: scripts
      })
      .then(response => {
            if (response.ok) {
                console.log('Message sent successfully!');
                return response.json();
              } else {
                console.error('Message sending failed!');
                // Handle error cases here
              }
      })
      .then(data => {
        // Handle the response data
        console.log('API response:', data.result);
        return data.result;
      })
      .catch(error => {
        console.error('Error sending request:', error);
      });

    
}