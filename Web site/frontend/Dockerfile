# Use lightweight Nginx
FROM nginx:alpine

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy all frontend files to the html directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80