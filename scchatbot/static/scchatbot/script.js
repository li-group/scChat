function embedVisualization(htmlContent) {
    // Create a div container instead of iframe for better PDF compatibility
    const container = document.createElement('div');
    container.style.border = 'none';
    container.style.width = '1200px';
    container.style.height = '800px';
    container.style.overflow = 'hidden';
    container.classList.add('plotly-container');
    
    // Create a temporary div to parse the HTML content
    const temp = document.createElement('div');
    temp.innerHTML = htmlContent;
    
    // Extract the plot div and scripts
    const plotDiv = temp.querySelector('div[id*="plotly"]') || temp.querySelector('div');
    const scripts = temp.querySelectorAll('script');
    
    if (plotDiv) {
        // Clone the plot div and add it to our container
        const clonedPlotDiv = plotDiv.cloneNode(true);
        container.appendChild(clonedPlotDiv);
        
        // Execute the scripts to render the plot
        scripts.forEach(script => {
            if (script.innerHTML && script.innerHTML.includes('Plotly')) {
                try {
                    // Create a new script element and execute it
                    const newScript = document.createElement('script');
                    newScript.innerHTML = script.innerHTML;
                    container.appendChild(newScript);
                } catch (error) {
                    console.error('Error executing Plotly script:', error);
                }
            }
        });
    } else {
        // Fallback: insert all content directly
        container.innerHTML = htmlContent;
        
        // Try to execute any scripts manually
        const allScripts = container.querySelectorAll('script');
        allScripts.forEach(script => {
            if (script.innerHTML) {
                try {
                    eval(script.innerHTML);
                } catch (error) {
                    console.error('Error executing script:', error);
                }
            }
        });
    }
    
    return container;
}

document.addEventListener("DOMContentLoaded", function() {
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages');
    const form = document.getElementById("chat-form");
    const fileInput = document.getElementById("file-upload");
    
    // WebSocket setup (optional - only if backend supports it)
    let chatSocket = null;
    let wsEnabled = false;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 3;
    // Use roomName from template, fallback to 'default' if not defined
    const wsRoomName = typeof roomName !== 'undefined' ? roomName : 'default';
    
    function connectWebSocket() {
        // Don't keep trying if we've failed too many times
        if (reconnectAttempts >= maxReconnectAttempts) {
            console.log('WebSocket disabled after max reconnection attempts');
            return;
        }
        
        try {
            const wsScheme = window.location.protocol === "https:" ? "wss" : "ws";
            chatSocket = new WebSocket(
                wsScheme + '://' + window.location.host + '/ws/chat/' + wsRoomName + '/'
            );
            
            chatSocket.onopen = function(e) {
                console.log('WebSocket connected successfully');
                wsEnabled = true;
                reconnectAttempts = 0;
            };
            
            chatSocket.onmessage = function(e) {
                const data = JSON.parse(e.data);
                
                if (data.type === 'progress') {
                    updateLoadingMessage(data.message, data.progress, data.stage);
                } else {
                    // Handle regular chat messages if needed
                    console.log('Received message:', data);
                }
            };
            
            chatSocket.onclose = function(e) {
                wsEnabled = false;
                reconnectAttempts++;
                
                if (reconnectAttempts < maxReconnectAttempts) {
                    console.log(`WebSocket closed. Reconnect attempt ${reconnectAttempts}/${maxReconnectAttempts} in 3 seconds...`);
                    setTimeout(() => {
                        connectWebSocket();
                    }, 3000);
                } else {
                    console.log('WebSocket support not available. Using basic loading spinner only.');
                }
            };
            
            chatSocket.onerror = function(e) {
                console.log('WebSocket connection failed. The server may not have WebSocket support enabled.');
            };
        } catch (error) {
            console.log('WebSocket not supported or not available:', error);
            wsEnabled = false;
        }
    }
    
    // Try to connect to WebSocket when page loads
    connectWebSocket();

    // Export chat as PDF without page breaks
    document.getElementById('export-pdf').addEventListener('click', async function() {
        const element = document.getElementById('messages-container');
        
        // Enhanced options for better plot capture
        const opt = {
            margin: [10, 10, 10, 10],
            filename: 'chat_history.pdf',
            image: { type: 'jpeg', quality: 0.98 },
            html2canvas: {
                scale: 2,
                useCORS: true,
                allowTaint: true,
                windowWidth: element.scrollWidth,
                windowHeight: element.scrollHeight,
                onclone: function(clonedDoc) {
                    // Ensure all Plotly plots are visible in the clone
                    const plotlyContainers = clonedDoc.querySelectorAll('.plotly-container');
                    plotlyContainers.forEach(container => {
                        container.style.visibility = 'visible';
                        container.style.display = 'block';
                    });
                }
            },
            jsPDF: {
                unit: 'px',
                format: [element.scrollWidth, element.scrollHeight],
                orientation: 'portrait',
            }
        };
        
        try {
            await html2pdf().set(opt).from(element).save();
        } catch (error) {
            console.error('PDF export failed:', error);
            alert('PDF export failed. Please try again.');
        }
    });

    // Export chat as text file
    document.getElementById('export-chat').addEventListener('click', function() {
        let chatHistory = '';
        const messages = messagesContainer.querySelectorAll('.user-message, .system-message');
        messages.forEach(function(message) {
            if (message.classList.contains('user-message')) {
                chatHistory += `User: ${message.innerText}\n`;
            } else if (message.classList.contains('system-message')) {
                chatHistory += `ScChat: ${message.innerText}\n`;
            }
        });
        const blob = new Blob([chatHistory], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    // Scroll to bottom function
    function scrollToBottom() {
        const lastMessage = messagesContainer.lastElementChild;
        if (lastMessage) {
            lastMessage.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }

    scrollToBottom();

    // Typewriter effect with text formatting
    // function typeWriter(text, element, callback) {
    //     let i = 0;
    //     const intervalId = setInterval(() => {
    //         if (i < text.length) {
    //             // Formatting logic (bold, italic, code, etc.)
    //             if (text.substring(i, i + 3) === '***') {
    //                 i += 3;
    //                 let boldItalicText = '';
    //                 while (i < text.length && text.substring(i, i + 3) !== '***') {
    //                     boldItalicText += text.charAt(i);
    //                     i++;
    //                 }
    //                 i += 3;
    //                 const boldItalicElement = document.createElement('strong');
    //                 const italicElement = document.createElement('em');
    //                 italicElement.textContent = boldItalicText;
    //                 boldItalicElement.appendChild(italicElement);
    //                 element.appendChild(boldItalicElement);
    //             } else if (text.substring(i, i + 2) === '**') {
    //                 i += 2;
    //                 let boldText = '';
    //                 while (i < text.length && text.substring(i, i + 2) !== '**') {
    //                     boldText += text.charAt(i);
    //                     i++;
    //                 }
    //                 i += 2;
    //                 const boldElement = document.createElement('strong');
    //                 boldElement.textContent = boldText;
    //                 element.appendChild(boldElement);
    //             } else if (text.charAt(i) === '*') {
    //                 i += 1;
    //                 let italicText = '';
    //                 while (i < text.length && text.charAt(i) !== '*') {
    //                     italicText += text.charAt(i);
    //                     i++;
    //                 }
    //                 i += 1;
    //                 const italicElement = document.createElement('em');
    //                 italicElement.textContent = italicText;
    //                 element.appendChild(italicElement);
    //             } else {
    //                 let char = text.charAt(i);
    //                 element.appendChild(document.createTextNode(char));
    //                 i++;
    //             }
    //             scrollToBottom();
    //         } else {
    //             clearInterval(intervalId);
    //             if (callback) callback();
    //         }
    //     }, 20);
    // }

    function typeWriter(text, element, callback) {
        let i = 0;
        const intervalId = setInterval(() => {
            if (i < text.length) {
                // Handle bold italic text (***...***)
                if (text.substring(i, i + 3) === '***') {
                    i += 3;
                    let boldItalicText = '';
                    while (i < text.length && text.substring(i, i + 3) !== '***') {
                        boldItalicText += text.charAt(i);
                        i++;
                    }
                    i += 3;
                    const boldItalicElement = document.createElement('strong');
                    const italicElement = document.createElement('em');
                    italicElement.textContent = boldItalicText;
                    boldItalicElement.appendChild(italicElement);
                    element.appendChild(boldItalicElement);
                }
                // Handle bold text (**...**)
                else if (text.substring(i, i + 2) === '**') {
                    i += 2;
                    let boldText = '';
                    while (i < text.length && text.substring(i, i + 2) !== '**') {
                        boldText += text.charAt(i);
                        i++;
                    }
                    i += 2;
                    const boldElement = document.createElement('strong');
                    boldElement.textContent = boldText;
                    element.appendChild(boldElement);
                }
                // Handle italic text (*...*)
                else if (text.charAt(i) === '*') {
                    i += 1;
                    let italicText = '';
                    while (i < text.length && text.charAt(i) !== '*') {
                        italicText += text.charAt(i);
                        i++;
                    }
                    i += 1;
                    const italicElement = document.createElement('em');
                    italicElement.textContent = italicText;
                    element.appendChild(italicElement);
                }
                // Handle strikethrough text (~~...~~)
                else if (text.substring(i, i + 2) === '~~') {
                    i += 2;
                    let strikethroughText = '';
                    while (i < text.length && text.substring(i, i + 2) !== '~~') {
                        strikethroughText += text.charAt(i);
                        i++;
                    }
                    i += 2;
                    const strikethroughElement = document.createElement('del');
                    strikethroughElement.textContent = strikethroughText;
                    element.appendChild(strikethroughElement);
                }
                // Handle inline code (`...`)
                else if (text.charAt(i) === '`') {
                    i++;
                    let inlineCodeText = '';
                    while (i < text.length && text.charAt(i) !== '`') {
                        inlineCodeText += text.charAt(i);
                        i++;
                    }
                    i++;
                    const codeElement = document.createElement('code');
                    codeElement.textContent = inlineCodeText;
                    element.appendChild(codeElement);
                }
                // Handle code blocks (```...```)
                else if (text.substring(i, i + 3) === '```') {
                    i += 3;
                    let codeBlockText = '';
                    while (i < text.length && text.substring(i, i + 3) !== '```') {
                        codeBlockText += text.charAt(i);
                        i++;
                    }
                    i += 3;
                    const codeBlockElement = document.createElement('pre');
                    const codeElement = document.createElement('code');
                    codeElement.textContent = codeBlockText;
                    codeBlockElement.appendChild(codeElement);
                    element.appendChild(codeBlockElement);
                }
                // Handle blockquotes (>...)
                else if (text.charAt(i) === '>') {
                    i++;
                    let blockquoteText = '';
                    while (i < text.length && text.charAt(i) !== '\n') {
                        blockquoteText += text.charAt(i);
                        i++;
                    }
                    const blockquoteElement = document.createElement('blockquote');
                    blockquoteElement.textContent = blockquoteText.trim();
                    element.appendChild(blockquoteElement);
                }
                // Handle headers (#, ##, ###)
                else if (text.substring(i, i + 3) === '###') {
                    i += 3;
                    let headerText = '';
                    while (i < text.length && text.charAt(i) !== '\n') {
                        headerText += text.charAt(i);
                        i++;
                    }
                    const headerElement = document.createElement('h3');
                    headerElement.textContent = headerText.trim();
                    element.appendChild(headerElement);
                }
                else if (text.substring(i, i + 2) === '##') {
                    i += 2;
                    let headerText = '';
                    while (i < text.length && text.charAt(i) !== '\n') {
                        headerText += text.charAt(i);
                        i++;
                    }
                    const headerElement = document.createElement('h2');
                    headerElement.textContent = headerText.trim();
                    element.appendChild(headerElement);
                }
                else if (text.charAt(i) === '#') {
                    i++;
                    let headerText = '';
                    while (i < text.length && text.charAt(i) !== '\n') {
                        headerText += text.charAt(i);
                        i++;
                    }
                    const headerElement = document.createElement('h1');
                    headerElement.textContent = headerText.trim();
                    element.appendChild(headerElement);
                }
                // Handle links ([text](url))
                else if (text.charAt(i) === '[') {
                    i++;
                    let linkText = '';
                    while (i < text.length && text.charAt(i) !== ']') {
                        linkText += text.charAt(i);
                        i++;
                    }
                    i += 2; // Skip "]("
                    let urlText = '';
                    while (i < text.length && text.charAt(i) !== ')') {
                        urlText += text.charAt(i);
                        i++;
                    }
                    i++; // Skip ")"
                    const linkElement = document.createElement('a');
                    linkElement.textContent = linkText;
                    linkElement.href = urlText;
                    linkElement.target = '_blank';
                    element.appendChild(linkElement);
                }
                // Handle images (![alt](url))
                else if (text.substring(i, i + 2) === '![') {
                    i += 2;
                    let altText = '';
                    while (i < text.length && text.charAt(i) !== ']') {
                        altText += text.charAt(i);
                        i++;
                    }
                    i += 2; // Skip "]("
                    let urlText = '';
                    while (i < text.length && text.charAt(i) !== ')') {
                        urlText += text.charAt(i);
                        i++;
                    }
                    i++; // Skip ")"
                    const imgElement = document.createElement('img');
                    imgElement.alt = altText;
                    imgElement.src = urlText;
                    imgElement.style.maxWidth = '100%';
                    element.appendChild(imgElement);
                }
                // Handle horizontal rule (---)
                else if (text.substring(i, i + 3) === '---') {
                    i += 3;
                    const hrElement = document.createElement('hr');
                    element.appendChild(hrElement);
                }
                // Handle line breaks
                else if (text.charAt(i) === '\n') {
                    element.appendChild(document.createElement('br'));
                    i++;
                }
                // Default case: append the character as is
                else {
                    let char = text.charAt(i);
                    element.appendChild(document.createTextNode(char));
                    i++;
                }
                scrollToBottom();
            } else {
                clearInterval(intervalId);
                if (callback) callback();
            }
        }, 20);
    }


    // // Adding messages to chat
    // function addChatMessage(messageText, isUser) {
    //     const messageElement = document.createElement('div');
    //     messageElement.classList.add(isUser ? 'user-message' : 'system-message');
    //     messagesContainer.appendChild(messageElement);

    //     if (isTextMessage(messageText)) {
    //         typeWriter(messageText, messageElement, () => {
    //             scrollToBottom();
    //         });
    //     } else {
    //         messageElement.textContent = messageText;
    //         scrollToBottom();
    //     }
    // }

    function addChatMessage(messageText, isUser, isHtml = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(isUser ? 'user-message' : 'system-message');
    
        // If isHtml is true, set innerHTML so the HTML is rendered; otherwise, use your typewriter/text approach.
        if (isHtml) {
            messageElement.innerHTML = messageText;
            scrollToBottom();
        } else {
            if (isTextMessage(messageText)) {
                typeWriter(messageText, messageElement, () => {
                    scrollToBottom();
                });
            } else {
                messageElement.textContent = messageText;
                scrollToBottom();
            }
        }
        messagesContainer.appendChild(messageElement);
    }
    
    let currentLoadingMessage = null;
    
    function addLoadingMessage() {
        const loadingElement = document.createElement('div');
        loadingElement.classList.add('system-message', 'loading-message');
        loadingElement.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <span class="loading-text">Processing your request...</span>
            </div>
        `;
        messagesContainer.appendChild(loadingElement);
        scrollToBottom();
        currentLoadingMessage = loadingElement;
        return loadingElement;
    }
    
    function updateLoadingMessage(message, progress, stage) {
        if (currentLoadingMessage) {
            const loadingText = currentLoadingMessage.querySelector('.loading-text');
            if (loadingText) {
                let updateText = message;
                if (stage) {
                    updateText = `[${stage}] ${message}`;
                }
                if (progress !== null && progress !== undefined) {
                    updateText += ` (${Math.round(progress)}%)`;
                }
                loadingText.textContent = updateText;
            }
        }
    }

    function isTextMessage(message) {
        return typeof message === 'string';
    }

    // Handling form submission
    if (form && messageInput && messagesContainer) {
        // form.onsubmit = async function(e) {
        //     e.preventDefault();
        //     const message = messageInput.value.trim();
        //     if (message) {
        //         messageInput.value = '';
        //         addChatMessage(message, true);
        //         scrollToBottom();

        //         try {
        //             const response = await fetch(chatUrl, {
        //                 method: "POST",
        //                 headers: {
        //                     "Content-Type": "application/json",
        //                     "X-CSRFToken": csrfToken
        //                 },
        //                 body: JSON.stringify({ message: message })
        //             });

        //             // const data = await response.json();
        //             // addChatMessage(data.response, false);

        //             // if (data.graph_json) {
        //             //     const graphJson = JSON.parse(data.graph_json);
        //             //     const graphDiv = document.createElement('div');
        //             //     messagesContainer.appendChild(graphDiv);
        //             //     Plotly.react(graphDiv, graphJson.data, graphJson.layout);
        //             //     scrollToBottom();
        //             // }
        //             const data = await response.json();
        //             console.log("Server response keys:", Object.keys(data));
        //             console.log("Server response (truncated):", JSON.stringify(data).substring(0,300) + "...");

        //             // If there is a simple text response
        //             if (data.response && data.response.trim() !== "") {
        //                 addChatMessage(data.response, false);
        //             }
                    
        //             // If the backend returned visualization HTML in "graph_html"
        //             if (data.graph_html) {
        //                 console.log("Embedding graph_html visualization...");
        //                 const iframe = embedVisualization(data.graph_html);
        //                 const container = document.createElement('div');
        //                 container.classList.add('system-message');  // use your assistant message class
        //                 container.appendChild(iframe);
        //                 messagesContainer.appendChild(container);
        //                 scrollToBottom();
        //             } 
        //             // Fallback for graph_json (if any)
        //             else if (data.graph_json) {
        //                 console.log("Rendering graph_json visualization...");
        //                 const graphJson = JSON.parse(data.graph_json);
        //                 const graphDiv = document.createElement('div');
        //                 messagesContainer.appendChild(graphDiv);
        //                 Plotly.react(graphDiv, graphJson.data, graphJson.layout);
        //                 scrollToBottom();
        //             }

        //         } catch (error) {
        //             console.error('Error:', error);
        //             addChatMessage("Failed to send message.", false);
        //             scrollToBottom();
        //         }
        //     }
        // };

        form.onsubmit = async function(e) {
            e.preventDefault();
            const message = messageInput.value.trim();
            if (message) {
                messageInput.value = '';
                addChatMessage(message, true);
                scrollToBottom();
                
                // Add loading indicator
                const loadingMessage = addLoadingMessage();
        
                try {
                    const response = await fetch(chatUrl, {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                            "X-CSRFToken": csrfToken
                        },
                        body: JSON.stringify({ message: message })
                    });
        
                    const data = await response.json();
                    
                    // Remove loading message
                    if (loadingMessage && loadingMessage.parentNode) {
                        loadingMessage.remove();
                    }
        
                    // NEW: Handle multiple messages structure (version 3.0)
                    if (data.response_type === "multiple_messages" && data.messages && Array.isArray(data.messages)) {
                        
                        // Render each message separately
                        data.messages.forEach((message, index) => {
                            if (message.message_type === "text") {
                                addChatMessage(message.response, false);
                            } else if (message.message_type === "plot") {
                                
                                // Create separate bot message container for this plot
                                const plotContainer = document.createElement('div');
                                plotContainer.classList.add('system-message', 'plot-section');
                                plotContainer.id = `plot-message-${index}`;
                                
                                // Add plot title/description header
                                const plotHeader = document.createElement('div');
                                plotHeader.classList.add('plot-header');
                                plotHeader.innerHTML = `<h4>${message.plot_title}</h4><p class="plot-description">${message.plot_description}</p>`;
                                plotContainer.appendChild(plotHeader);
                                
                                // Create iframe for this plot
                                const iframe = embedVisualization(message.graph_html);
                                iframe.id = `iframe-plot-${index}`;
                                iframe.style.height = '650px';
                                
                                plotContainer.appendChild(iframe);
                                messagesContainer.appendChild(plotContainer);
                            }
                        });
                        
                        scrollToBottom();
                        
                    } 
                    // FALLBACK: Handle legacy individual plots structure (version 2.0)
                    else if (data.plots && Array.isArray(data.plots) && data.plots.length > 0) {
                        
                        // If there is a simple text response, display it first
                        if (data.response && data.response.trim() !== "") {
                            addChatMessage(data.response, false);
                        }
                        
                        // Render each plot in its own section
                        data.plots.forEach((plot, index) => {
                            
                            // Create container for this specific plot
                            const plotContainer = document.createElement('div');
                            plotContainer.classList.add('system-message', 'plot-section');
                            plotContainer.id = `plot-section-${plot.id}`;
                            
                            // Add plot title/description header
                            const plotHeader = document.createElement('div');
                            plotHeader.classList.add('plot-header');
                            plotHeader.innerHTML = `<h4>${plot.title}</h4><p class="plot-description">${plot.description}</p>`;
                            plotContainer.appendChild(plotHeader);
                            
                            // Create iframe for this specific plot
                            const iframe = embedVisualization(plot.html);
                            iframe.id = `iframe-${plot.id}`;
                            iframe.style.height = '650px'; // Slightly larger for individual plots
                            
                            plotContainer.appendChild(iframe);
                            messagesContainer.appendChild(plotContainer);
                        });
                        
                        scrollToBottom();
                        
                    } 
                    // Handle single text response (no plots)
                    else if (data.response && data.response.trim() !== "") {
                        addChatMessage(data.response, false);
                    }
                    // FALLBACK: Handle legacy combined plots (version 1.0)
                    else if (data.graph_html) {
                        // Fallback to legacy combined plots for backward compatibility
                        const iframe = embedVisualization(data.graph_html);
                        
                        const container = document.createElement('div');
                        container.classList.add('system-message');
                        container.appendChild(iframe);
                        messagesContainer.appendChild(container);
                        
                        scrollToBottom();
                    } else if (data.graph_json) {
                        const graphJson = JSON.parse(data.graph_json);
                        const graphDiv = document.createElement('div');
                        messagesContainer.appendChild(graphDiv);
                        Plotly.react(graphDiv, graphJson.data, graphJson.layout);
                        scrollToBottom();
                    }
                    
                } catch (error) {
                    console.error('Error:', error);
                    // Remove loading message on error
                    if (loadingMessage && loadingMessage.parentNode) {
                        loadingMessage.remove();
                    }
                    addChatMessage("Failed to send message.", false);
                    scrollToBottom();
                }
            }
        };
    }

    // Handling file uploads
    // fileInput.addEventListener('change', function(event) {
    //     if (this.files.length > 0) {
    //         const file = this.files[0];
    //         readFileContent(file);
    //         const formData = new FormData();
    //         const uploadUrl = 'upload/';  
    //         formData.append('file', file, file.name);

    //         fetch(uploadUrl, {
    //             method: 'POST',
    //             body: formData,
    //             headers: {
    //                 "X-CSRFToken": csrfToken
    //             },
    //         })
    //         .then(response => response.json())
    //         .then(data => {
    //             addChatMessage(data.response, false);
    //             scrollToBottom();
    //         })
    //         .catch(error => {
    //             console.error('Error:', error);
    //             addChatMessage("Failed to send message.", false);
    //             scrollToBottom();
    //         });
    //     }
    // });

    // Handling file uploads
    fileInput.addEventListener('change', function(event) {
        if (this.files.length > 0) {
            const file = this.files[0];
            readFileContent(file);
            const formData = new FormData();
            const uploadUrl = 'upload/';  
            formData.append('file', file, file.name);

            fetch(uploadUrl, {
                method: 'POST',
                body: formData,
                headers: {
                    "X-CSRFToken": csrfToken
                },
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    console.log(data.message);
                } else {
                    console.error('Upload failed:', data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    });

    
    function readFileContent(file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                displayFileInChat(`<img src="${e.target.result}" alt="Uploaded Image" style="max-width: 100%; height: auto;">`);
                scrollToBottom();
            };
            reader.readAsDataURL(file);
        } else {
            displayFileInChat(`File uploaded: ${file.name}`);
            scrollToBottom();
        }
    }

    // // Reading file content
    // function readFileContent(file) {
    //     const reader = new FileReader();

    //     if (file.type.startsWith('image/')) {
    //         reader.onload = function(e) {
    //             displayFileInChat(`<img src="${e.target.result}" alt="Uploaded Image" style="max-width: 100%; height: auto;">`);
    //             scrollToBottom();
    //         };
    //         reader.readAsDataURL(file);
    //     } else if (file.type.startsWith('text/')) {
    //         reader.onload = function(e) {
    //             displayFileInChat(`<pre>${e.target.result}</pre>`);
    //             scrollToBottom();
    //         };
    //         reader.readAsText(file);
    //     } else if (file.type === 'application/pdf' || file.name.endsWith('.h5ad')) {
    //         displayFileInChat(`File uploaded: ${file.name}`);
    //         scrollToBottom();
    //     } else {
    //         displayFileInChat("Unsupported file type.");
    //         scrollToBottom();
    //     }
    // }

    // Displaying file in chat
    function displayFileInChat(content) {
        const messageElement = document.createElement("div");
        messageElement.innerHTML = content;
        messageElement.classList.add('system-message');
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
});
