document.addEventListener("DOMContentLoaded", function() {
    const messageInput = document.getElementById('message-input');
    const messagesContainer = document.getElementById('messages');
    const form = document.getElementById("chat-form");
    const fileInput = document.getElementById("file-upload");

    // Export chat as PDF without page breaks
    document.getElementById('export-pdf').addEventListener('click', function() {
        const element = document.getElementById('messages-container');
        const opt = {
            margin: [0, 0, 0, 0],
            filename: 'chat_history.pdf',
            image: { type: 'jpeg', quality: 1.0 },
            html2canvas: {
                scale: 2,
                windowWidth: element.scrollWidth,
                windowHeight: element.scrollHeight,
            },
            jsPDF: {
                unit: 'px',
                format: [element.scrollWidth, element.scrollHeight],
                orientation: 'portrait',
            }
        };
        html2pdf().set(opt).from(element).save();
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


    // Adding messages to chat
    function addChatMessage(messageText, isUser) {
        const messageElement = document.createElement('div');
        messageElement.classList.add(isUser ? 'user-message' : 'system-message');
        messagesContainer.appendChild(messageElement);

        if (isTextMessage(messageText)) {
            typeWriter(messageText, messageElement, () => {
                scrollToBottom();
            });
        } else {
            messageElement.textContent = messageText;
            scrollToBottom();
        }
    }

    function isTextMessage(message) {
        return typeof message === 'string';
    }

    // Handling form submission
    if (form && messageInput && messagesContainer) {
        form.onsubmit = async function(e) {
            e.preventDefault();
            const message = messageInput.value.trim();
            if (message) {
                messageInput.value = '';
                addChatMessage(message, true);
                scrollToBottom();

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
                    addChatMessage(data.response, false);

                    if (data.graph_json) {
                        const graphJson = JSON.parse(data.graph_json);
                        const graphDiv = document.createElement('div');
                        messagesContainer.appendChild(graphDiv);
                        Plotly.react(graphDiv, graphJson.data, graphJson.layout);
                        scrollToBottom();
                    }
                } catch (error) {
                    console.error('Error:', error);
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
