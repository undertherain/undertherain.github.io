#./themes/blowfish/node_modules/tailwindcss/lib/cli.js -c tailwind.config.js  -o my.css
#cat my.css | grep indigo
#dev
#NODE_ENV=development ./themes/blowfish/node_modules/tailwindcss/lib/cli.js -c ./themes/blowfish/tailwind.config.js -i ./themes/blowfish/assets/css/main.css -o ./assets/css/compiled/main.css --jit -w",
#build
NODE_ENV=production \
    npx ./themes/blowfish/node_modules/@tailwindcss/cli \
    -i ./assets/css/main.css \
    -c ./themes/blowfish/tailwind.config.js \
    -o ./assets/css/compiled/main.css --jit
#-i "./themes/blowfish/assets/css/main.css  ./static/css/quiz.css" \
#-i ./static/css/quiz.css \
# TODO: modify theme's config to add additional colors