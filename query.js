const phone = '17069310291'
const arr = [
    '17069310291',
    '18736805971'
]

isPhoneInArray()

function isPhoneInArray() {
    let result = arr.includes(phone)
    console.log('结果：', result)
}