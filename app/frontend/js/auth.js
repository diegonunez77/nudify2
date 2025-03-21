// Google Sign-In Authentication Module

// Store the current user information
let currentUser = null;

// Initialize Google Sign-In
function initGoogleAuth() {
    // Load the Google Sign-In API script
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);

    script.onload = () => {
        // Initialize Google Sign-In button
        google.accounts.id.initialize({
            client_id: getGoogleClientId(),
            callback: handleGoogleSignIn,
            auto_select: false,
            cancel_on_tap_outside: true
        });

        // Render the Google Sign-In button
        renderGoogleSignInButton();
    };
}

// Get Google Client ID from the backend
function getGoogleClientId() {
    // This should be replaced with your actual Google Client ID
    // For production, you should fetch this from the backend
    return 'YOUR_GOOGLE_CLIENT_ID';
}

// Render Google Sign-In button
function renderGoogleSignInButton() {
    // Render the button in the auth-container
    const authContainer = document.getElementById('auth-container');
    if (authContainer) {
        google.accounts.id.renderButton(
            authContainer,
            { 
                theme: 'outline', 
                size: 'large',
                text: 'signin_with',
                shape: 'rectangular',
                logo_alignment: 'left'
            }
        );
    }
}

// Handle Google Sign-In
async function handleGoogleSignIn(response) {
    try {
        // Send the ID token to the backend
        const result = await fetch('/auth/google', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id_token: response.credential
            })
        });

        const data = await result.json();

        if (data.success) {
            // Store user data
            currentUser = data.user;
            
            // Update UI based on authentication
            updateAuthUI();
            
            // Show success notification
            showNotification('Successfully signed in!', 'success');
            
            // Refresh credit display
            updateCreditDisplay();
        } else {
            console.error('Authentication failed:', data.error);
            showNotification('Authentication failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error during authentication:', error);
        showNotification('Error during authentication. Please try again.', 'error');
    }
}

// Sign out the current user
async function signOut() {
    try {
        const response = await fetch('/auth/logout', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (data.success) {
            // Clear user data
            currentUser = null;
            
            // Update UI based on authentication
            updateAuthUI();
            
            // Show success notification
            showNotification('Successfully signed out!', 'success');
        } else {
            console.error('Sign out failed:', data.error);
            showNotification('Sign out failed: ' + data.error, 'error');
        }
    } catch (error) {
        console.error('Error during sign out:', error);
        showNotification('Error during sign out. Please try again.', 'error');
    }
}

// Check if user is authenticated
async function checkAuthentication() {
    try {
        const response = await fetch('/auth/user', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();

        if (data.user) {
            // Store user data
            currentUser = data.user;
            
            // Update UI based on authentication
            updateAuthUI();
            
            // Update credit display
            updateCreditDisplay();
            
            return true;
        } else {
            // Clear user data
            currentUser = null;
            
            // Update UI based on authentication
            updateAuthUI();
            
            return false;
        }
    } catch (error) {
        console.error('Error checking authentication:', error);
        // Clear user data
        currentUser = null;
        
        // Update UI based on authentication
        updateAuthUI();
        
        return false;
    }
}

// Update UI based on authentication status
function updateAuthUI() {
    const authContainer = document.getElementById('auth-container');
    const userInfoContainer = document.getElementById('user-info');
    const imageUploadSection = document.getElementById('image-upload-section');
    const imageUrlSection = document.getElementById('image-url-section');
    const loginRequiredMessage = document.getElementById('login-required-message');
    
    if (currentUser) {
        // User is authenticated
        if (authContainer) authContainer.style.display = 'none';
        
        // Show user info
        if (userInfoContainer) {
            userInfoContainer.style.display = 'flex';
            userInfoContainer.innerHTML = `
                <div class="user-profile">
                    ${currentUser.profile_picture ? `<img src="${currentUser.profile_picture}" alt="Profile" class="profile-picture">` : ''}
                    <div class="user-details">
                        <span class="user-name">${currentUser.display_name || currentUser.email}</span>
                        <span class="user-credits">Credits: <span id="credit-balance">${currentUser.credit_balance}</span></span>
                    </div>
                </div>
                <button id="sign-out-button" class="btn btn-outline-danger">Sign Out</button>
            `;
            
            // Add event listener to sign out button
            document.getElementById('sign-out-button').addEventListener('click', signOut);
        }
        
        // Show image upload and URL sections
        if (imageUploadSection) imageUploadSection.style.display = 'block';
        if (imageUrlSection) imageUrlSection.style.display = 'block';
        
        // Hide login required message
        if (loginRequiredMessage) loginRequiredMessage.style.display = 'none';
    } else {
        // User is not authenticated
        if (authContainer) authContainer.style.display = 'block';
        
        // Hide user info
        if (userInfoContainer) userInfoContainer.style.display = 'none';
        
        // Hide image upload and URL sections
        if (imageUploadSection) imageUploadSection.style.display = 'none';
        if (imageUrlSection) imageUrlSection.style.display = 'none';
        
        // Show login required message
        if (loginRequiredMessage) loginRequiredMessage.style.display = 'block';
    }
}

// Update credit display
async function updateCreditDisplay() {
    if (!currentUser) return;
    
    try {
        const response = await fetch('/api/user/credits', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();
        
        // Update credit balance display
        const creditBalanceElement = document.getElementById('credit-balance');
        if (creditBalanceElement && data.credit_balance !== undefined) {
            creditBalanceElement.textContent = data.credit_balance;
            currentUser.credit_balance = data.credit_balance;
        }
    } catch (error) {
        console.error('Error fetching credit balance:', error);
    }
}

// Show notification
function showNotification(message, type = 'info') {
    const notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) return;
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add notification to container
    notificationContainer.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        notification.classList.add('fade-out');
        setTimeout(() => {
            notificationContainer.removeChild(notification);
        }, 500);
    }, 5000);
}

// Initialize authentication when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Google Sign-In
    initGoogleAuth();
    
    // Check if user is already authenticated
    checkAuthentication();
});

// Export functions for use in other modules
window.auth = {
    signOut,
    checkAuthentication,
    updateCreditDisplay,
    getCurrentUser: () => currentUser,
    isAuthenticated: () => !!currentUser
};
